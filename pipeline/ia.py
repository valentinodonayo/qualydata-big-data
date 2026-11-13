"""Etapa de IA: un modelo de lenguaje decide qué bando "apoya" cada narrativa y le pone título.

Proveedores (se usa el primero que tenga clave):
1. Google Gemini Flash (GEMINI_API_KEY): tiene capa GRATIS, sin tarjeta. Es el recomendado.
2. Claude Haiku (ANTHROPIC_API_KEY): pago, muy barato.
Sin ninguna clave: se usa la caché versionada y, si falta algo, el respaldo sin IA.

Por qué usamos IA acá y no en todo:
- Es UNA llamada por narrativa (~13), no una por comentario (150.000).
  Cuesta menos de 1 centavo por corrida.
- Es una decisión que necesita "entender": ¿cuál de los dos bandos sostiene
  la afirmación y cuál la niega? Ninguna regla simple acertó siempre.

Las claves se leen del entorno o del archivo .env, que NUNCA se sube al repositorio.

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
import time
import urllib.error
import urllib.request
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

# --- Gemini (Google). Modelo estable con capa gratis (ver aistudio.google.com/docs/pricing) ---
MODELO_GEMINI = "gemini-3.8-flash"
URL_GEMINI = "https://generativelanguage.googleapis.com/v1beta/models/{modelo}:generateContent"
MAX_TOKENS_GEMINI = 8192   # holgado: en Gemini el "razonamiento" interno también gasta tokens de salida
PARAMETROS_GEMINI = {"max_tokens": MAX_TOKENS_GEMINI, "formato": "json", "esquema": ESQUEMA}
# En la capa gratis no se paga nada: el costo real es US$ 0 (los tokens igual se anotan)
PRECIO_GEMINI_ENTRADA = 0.0
PRECIO_GEMINI_SALIDA = 0.0

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


def _cargar_clave(nombre="ANTHROPIC_API_KEY"):
    """Busca la clave `nombre` en el entorno o en el archivo .env (sin mostrarla nunca).
    Ignora el texto de ejemplo "poner-tu-clave-aca" de .env.example."""
    valor = os.environ.get(nombre, "")
    if not valor:
        ruta = Path(__file__).resolve().parent.parent / ".env"
        if ruta.exists():
            for linea in ruta.read_text(encoding="utf-8").splitlines():
                if linea.strip().startswith(nombre + "="):
                    valor = linea.split("=", 1)[1].strip().strip('"').strip("'")
    if not valor or "poner-tu-clave" in valor:
        return False
    os.environ[nombre] = valor
    return True


def _elegir_proveedor():
    """Devuelve el proveedor a usar según qué clave haya: "gemini", "claude" o None."""
    if _cargar_clave("GEMINI_API_KEY"):
        return "gemini"
    if _cargar_clave("ANTHROPIC_API_KEY"):
        return "claude"
    return None


def _config(proveedor):
    """(modelo, parámetros) del proveedor: entran en la huella de la caché."""
    if proveedor == "gemini":
        return MODELO_GEMINI, PARAMETROS_GEMINI
    return MODELO, PARAMETROS


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


def _cuerpos_gemini(mensaje):
    """Formas del pedido JSON a Gemini, de la más nueva a la más vieja.
    La documentación cambió de nombre los campos varias veces: si una forma da error 400,
    se prueba la siguiente (todas piden lo mismo: responder con JSON según ESQUEMA)."""
    base = {"systemInstruction": {"parts": [{"text": INSTRUCCIONES}]},
            "contents": [{"role": "user", "parts": [{"text": mensaje}]}]}
    return [
        {**base, "generationConfig": {"maxOutputTokens": MAX_TOKENS_GEMINI,
                                      "responseFormat": {"text": {"mimeType": "APPLICATION_JSON", "schema": ESQUEMA}}}},
        {**base, "generationConfig": {"maxOutputTokens": MAX_TOKENS_GEMINI,
                                      "responseMimeType": "application/json", "responseJsonSchema": ESQUEMA}},
        {**base, "generationConfig": {"maxOutputTokens": MAX_TOKENS_GEMINI,
                                      "responseMimeType": "application/json"}},
    ]


def _post_gemini(cuerpo):
    """Un POST a la API de Gemini con la librería estándar (sin dependencias nuevas).
    Devuelve (código HTTP, json o texto del error)."""
    pedido = urllib.request.Request(
        URL_GEMINI.format(modelo=MODELO_GEMINI), data=json.dumps(cuerpo).encode("utf-8"), method="POST",
        headers={"Content-Type": "application/json", "x-goog-api-key": os.environ["GEMINI_API_KEY"]})
    try:
        with urllib.request.urlopen(pedido, timeout=90) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8", "replace")[:300]


def _preguntar_gemini(etiqueta, mensaje):
    """Una llamada a Gemini. Devuelve la entrada para la caché, o None si no hubo respuesta usable."""
    for cuerpo in _cuerpos_gemini(mensaje):
        for intento in range(4):
            try:
                codigo, datos = _post_gemini(cuerpo)
            except Exception as error:      # sin internet, tiempo agotado...
                logging.warning("IA: falló la narrativa %s (%s)", etiqueta, type(error).__name__)
                return None
            if codigo in (429, 500, 503):   # demasiados pedidos (capa gratis) o servidor ocupado: esperar
                time.sleep(15 * (intento + 1))
                continue
            break
        if codigo == 400:                   # forma del pedido no aceptada: probar la siguiente
            continue
        if codigo != 200:
            # El texto del error de Google nunca incluye la clave; ayuda a entender qué pasó
            logging.warning("IA: falló la narrativa %s (HTTP %s: %s)", etiqueta, codigo, str(datos)[:160])
            return None
        partes = (datos.get("candidates") or [{}])[0].get("content", {}).get("parts", [])
        texto = "".join(p.get("text", "") for p in partes if not p.get("thought"))
        try:
            respuesta = json.loads(texto)
        except json.JSONDecodeError:
            logging.warning("IA: Gemini no devolvió JSON válido para %s", etiqueta)
            return None
        if respuesta.get("bando_que_afirma") not in ("A", "B"):
            return None
        uso = datos.get("usageMetadata", {})
        entrada = uso.get("promptTokenCount", 0)
        salida = uso.get("candidatesTokenCount", 0) + uso.get("thoughtsTokenCount", 0)
        return {"respuesta": respuesta, "tokens_entrada": entrada, "tokens_salida": salida,
                "usd": round(entrada * PRECIO_GEMINI_ENTRADA + salida * PRECIO_GEMINI_SALIDA, 8)}
    logging.warning("IA: Gemini rechazó todas las formas del pedido para %s", etiqueta)
    return None


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
    # Proveedor: el que tenga clave. Sin clave, la huella es la de Gemini si la caché tiene
    # respuestas de Gemini (así el jurado reproduce la entrega), y si no, la de Claude.
    proveedor = _elegir_proveedor()
    if proveedor is None:
        proveedor = "gemini" if any(e.get("modelo") == MODELO_GEMINI for e in cache.datos.values()) else "claude"
    modelo, parametros = _config(proveedor)
    claves = {k: cache_ia.huella(modelo, parametros, INSTRUCCIONES, m) for k, m in mensajes.items()}
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

    if faltan and _elegir_proveedor():
        if proveedor == "gemini":
            # De a una: la capa gratis limita los pedidos por minuto
            resultados = {k: _preguntar_gemini(k, mensajes[k]) for k in faltan}
        else:
            import anthropic
            cliente = anthropic.Anthropic(max_retries=3, timeout=60)
            # Hasta 8 narrativas en paralelo para no esperar una por una
            with ThreadPoolExecutor(max_workers=8) as pool:
                futuros = {k: pool.submit(_preguntar, cliente, k, mensajes[k]) for k in faltan}
            resultados = {k: f.result() for k, f in futuros.items()}
        for k, entrada in resultados.items():
            if entrada:
                entrada.update({"etiqueta": k, "huella_datos": huella_datos, "ejemplos": resumen[k]["ejemplos"],
                                "modelo": modelo})
                cache.guardar(claves[k], entrada)
                entradas[k] = entrada
        cache.escribir()
    elif faltan:
        logging.warning("IA: %s narrativas sin respuesta en la caché y sin clave (GEMINI_API_KEY ni ANTHROPIC_API_KEY): "
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
    crono.ia_modelo = modelo   # para que costo.json diga qué modelo se usó
    logging.info("IA: %s de %s narrativas resueltas con %s (%s de la caché, %s nuevas; US$ %.4f)",
                 len(titulos), len(resumen), modelo, cache.aciertos, cache.nuevas, crono.usd_ia)
    return apoya_es, titulos
