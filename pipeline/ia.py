"""Etapa de IA: Claude Haiku decide qué bando "apoya" cada narrativa y le pone título.

Por qué usamos IA acá y no en todo:
- Es UNA llamada por narrativa (~13), no una por comentario (150.000).
  Cuesta menos de 1 centavo por corrida.
- Es una decisión que necesita "entender": ¿cuál de los dos bandos sostiene
  la afirmación y cuál la niega? Ninguna regla simple acertó siempre.

La clave (ANTHROPIC_API_KEY) se lee del entorno o del archivo .env, que NUNCA
se sube al repositorio.

Reproducibilidad: cada respuesta se guarda en una caché versionada (modelos/cache_ia.json,
ver cache_ia.py). Con la caché, el pipeline reproduce la entrega SIN clave. Si falta una
respuesta y no hay clave (o la llamada falla), esa narrativa usa el respaldo sin IA:
"el bando más grande es el que apoya".

Haiku 5.5 no acepta temperature (la API devuelve error con valores no estándar): la
caché es lo que hace la salida idéntica entre corridas.
"""
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pipeline import cache_ia

MODELO = "claude-haiku-5-5"
PRECIO_ENTRADA = 0.10 / 1_000_000   # US$ por token de entrada (Haiku 5.5)
PRECIO_SALIDA = 0.50 / 1_000_000    # US$ por token de salida

ESQUEMA = {
    "type": "object",
    "properties": {
        "afirmacion": {"type": "string"},
        "bando_que_afirma": {"type": "string", "enum": ["A", "B"]},
        "titulo": {"type": "string"},
        "frases_clave": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["afirmacion", "bando_que_afirma", "titulo", "frases_clave"],
    "additionalProperties": False,
}

# Parámetros del pedido (entran en la huella de la caché: si cambian, se vuelve a preguntar)
MAX_TOKENS = 1024
PARAMETROS = {"max_tokens": MAX_TOKENS, "thinking": "disabled", "effort": "low", "esquema": ESQUEMA}

INSTRUCCIONES = """Analizás comentarios de redes sociales en español rioplatense.
Te paso comentarios de UNA misma discusión, separados en dos bandos opuestos (A y B).
Una "narrativa" es la afirmación concreta que circula: un reclamo, una denuncia,
un rumor o una propuesta (por ejemplo "la obra está parada porque se robaron la plata").
Un bando la AFIRMA y el otro la NIEGA, la refuta o la relativiza.

Respondé:
- afirmacion: la afirmación central, en una frase.
- bando_que_afirma: "A" o "B", el bando que sostiene esa afirmación.
- titulo: la afirmación como título corto (máximo 12 palabras), en español.
- frases_clave: 4 a 6 palabras o frases cortas típicas de la narrativa."""


def _cargar_clave():
    """Busca ANTHROPIC_API_KEY en el entorno o en el archivo .env (sin mostrarla nunca)."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return True
    ruta = Path(__file__).resolve().parent.parent / ".env"
    if ruta.exists():
        for linea in ruta.read_text(encoding="utf-8").splitlines():
            if linea.strip().startswith("ANTHROPIC_API_KEY="):
                os.environ["ANTHROPIC_API_KEY"] = linea.split("=", 1)[1].strip().strip('"').strip("'")
                return True
    return False


def _mensaje(info):
    """Arma el mensaje con los ejemplos de los dos bandos (siempre igual para los mismos datos)."""
    bloques = []
    for nombre, ejemplos in zip("AB", info["ejemplos"]):
        lista = "\n".join(f"- {t}" for t in ejemplos)
        bloques.append(f"BANDO {nombre}:\n{lista}")
    return "\n\n".join(bloques)


def _preguntar(cliente, etiqueta, mensaje):
    """Una llamada a Haiku. Devuelve la entrada para la caché, o None si no hubo respuesta usable."""
    try:
        respuesta = cliente.messages.create(
            model=MODELO,
            max_tokens=MAX_TOKENS,
            system=INSTRUCCIONES,
            # Tarea simple: sin razonamiento extendido y con poco esfuerzo (más barato y rápido)
            thinking={"type": "disabled"},
            output_config={"effort": "low",
                           "format": {"type": "json_schema", "schema": ESQUEMA}},
            messages=[{"role": "user", "content": mensaje}],
        )
    except Exception as error:   # red, clave inválida, límite de uso...: seguimos con el respaldo
        logging.warning("IA: falló la narrativa %s (%s)", etiqueta, type(error).__name__)
        return None
    uso = respuesta.usage
    if respuesta.stop_reason == "refusal":
        logging.warning("IA: Haiku no quiso responder sobre %s; uso el respaldo", etiqueta)
        return None
    texto = next((b.text for b in respuesta.content if b.type == "text"), "")
    try:
        datos = json.loads(texto)
    except json.JSONDecodeError:
        return None
    return {"respuesta": datos, "tokens_entrada": uso.input_tokens, "tokens_salida": uso.output_tokens,
            "usd": round(uso.input_tokens * PRECIO_ENTRADA + uso.output_tokens * PRECIO_SALIDA, 8)}


MIN_COINCIDENCIAS = 4   # respaldo difuso: ejemplos en común (de 12) para reconocer un bando


def _buscar_parecida(cache, huella_datos, etiqueta, ejemplos):
    """Si no hay coincidencia exacta: busca una respuesta guardada del MISMO set de datos y la
    misma narrativa, y reconoce cada bando por sus ejemplos en común (los ejemplos son los más
    centrales, así que casi no cambian entre máquinas). Devuelve (entrada, invertida) o (None, False).
    """
    mejor, invertida, mejor_puntaje = None, False, 0
    for entrada in cache.datos.values():
        if entrada.get("huella_datos") != huella_datos or entrada.get("etiqueta") != etiqueta:
            continue
        a, b = (set(x) for x in entrada.get("ejemplos", [[], []]))
        directo = len(a & set(ejemplos[0])) + len(b & set(ejemplos[1]))
        cruzado = len(a & set(ejemplos[1])) + len(b & set(ejemplos[0]))
        if max(directo, cruzado) >= 2 * MIN_COINCIDENCIAS and max(directo, cruzado) > mejor_puntaje \
                and directo != cruzado:
            mejor, invertida, mejor_puntaje = entrada, cruzado > directo, max(directo, cruzado)
    return mejor, invertida


def decidir(resumen, crono, ruta_cache=None, huella_datos=""):
    """Para cada narrativa, devuelve (apoya_es, titulos).

    apoya_es: {narrativa: 0 o 1}  qué bando es el que apoya
    titulos:  {narrativa: {"titulo": ..., "frases_clave": [...]}}  (solo si hubo respuesta de la IA)

    Orden: 1) caché versionada, 2) modelo (si hay clave), 3) respaldo sin IA.
    El costo que se suma a costo.json es el de la corrida que generó cada respuesta.
    """
    # Respaldo: el bando más grande es el que apoya
    apoya_es = {k: (0 if v["tamano"][0] >= v["tamano"][1] else 1) for k, v in resumen.items()}
    titulos = {}
    cache = cache_ia.CacheIA(ruta_cache) if ruta_cache else cache_ia.CacheIA()
    mensajes = {k: _mensaje(v) for k, v in resumen.items()}
    claves = {k: cache_ia.huella(MODELO, PARAMETROS, INSTRUCCIONES, m) for k, m in mensajes.items()}
    entradas = {k: cache.buscar(c) for k, c in claves.items()}
    invertidas = set()
    for k in sorted(k for k, e in entradas.items() if e is None):
        parecida, invertida = _buscar_parecida(cache, huella_datos, k, resumen[k]["ejemplos"])
        if parecida:
            entradas[k] = parecida
            if invertida:
                invertidas.add(k)
            logging.info("IA: %s resuelta con una respuesta guardada parecida (ejemplos en común)", k)
    faltan = sorted(k for k, e in entradas.items() if e is None)

    if faltan and _cargar_clave():
        import anthropic
        cliente = anthropic.Anthropic(max_retries=3, timeout=60)
        # Hasta 8 narrativas en paralelo para no esperar una por una
        with ThreadPoolExecutor(max_workers=8) as pool:
            futuros = {k: pool.submit(_preguntar, cliente, k, mensajes[k]) for k in faltan}
        for k, futuro in futuros.items():
            entrada = futuro.result()
            if entrada:
                entrada.update({"etiqueta": k, "huella_datos": huella_datos, "ejemplos": resumen[k]["ejemplos"]})
                cache.guardar(claves[k], entrada)
                entradas[k] = entrada
        cache.escribir()
    elif faltan:
        logging.warning("IA: %s narrativas sin respuesta en la caché y sin ANTHROPIC_API_KEY: "
                        "uso el respaldo (el bando más grande apoya)", len(faltan))

    for k in sorted(entradas):
        e = entradas[k]
        if not e:
            continue
        crono.sumar_ia(e["tokens_entrada"] + e["tokens_salida"], e["usd"])
        r = e["respuesta"]
        apoya_es[k] = 0 if r["bando_que_afirma"] == "A" else 1
        if k in invertidas:      # la respuesta guardada tenía los bandos al revés
            apoya_es[k] = 1 - apoya_es[k]
        titulos[k] = {"titulo": r["titulo"].strip(), "frases_clave": [f.strip() for f in r["frases_clave"]]}
    crono.ia_desde_cache = cache.aciertos
    crono.ia_nuevas = cache.nuevas
    logging.info("IA: %s de %s narrativas resueltas con %s (%s de la caché, %s nuevas; US$ %.4f)",
                 len(titulos), len(resumen), MODELO, cache.aciertos, cache.nuevas, crono.usd_ia)
    return apoya_es, titulos
