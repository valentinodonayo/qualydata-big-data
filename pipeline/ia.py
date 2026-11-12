"""Etapa de IA: Claude Haiku decide qué bando "apoya" cada narrativa y le pone título.

Por qué usamos IA acá y no en todo:
- Es UNA llamada por narrativa (~13), no una por comentario (150.000).
  Cuesta menos de 1 centavo por corrida.
- Es una decisión que necesita "entender": ¿cuál de los dos bandos sostiene
  la afirmación y cuál la niega? Ninguna regla simple acertó siempre.

La clave (ANTHROPIC_API_KEY) se lee del entorno o del archivo .env, que NUNCA
se sube al repositorio. Si no hay clave o la llamada falla, el pipeline sigue
igual con un respaldo: "el bando más grande es el que apoya".
"""
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

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


def _preguntar(cliente, etiqueta, info, crono):
    """Una llamada a Haiku para una narrativa. Devuelve el JSON de respuesta o None."""
    bloques = []
    for nombre, ejemplos in zip("AB", info["ejemplos"]):
        lista = "\n".join(f"- {t}" for t in ejemplos)
        bloques.append(f"BANDO {nombre}:\n{lista}")
    try:
        respuesta = cliente.messages.create(
            model=MODELO,
            max_tokens=1024,
            system=INSTRUCCIONES,
            # Tarea simple: pensar poco (más barato y rápido)
            thinking={"type": "disabled"},
            output_config={"effort": "low",
                           "format": {"type": "json_schema", "schema": ESQUEMA}},
            messages=[{"role": "user", "content": "\n\n".join(bloques)}],
        )
    except Exception as error:   # red, clave inválida, límite de uso...: seguimos con el respaldo
        logging.warning("IA: falló la narrativa %s (%s)", etiqueta, type(error).__name__)
        return None
    uso = respuesta.usage
    crono.sumar_ia(uso.input_tokens + uso.output_tokens,
                   uso.input_tokens * PRECIO_ENTRADA + uso.output_tokens * PRECIO_SALIDA)
    if respuesta.stop_reason == "refusal":
        logging.warning("IA: Haiku no quiso responder sobre %s; uso el respaldo", etiqueta)
        return None
    texto = next((b.text for b in respuesta.content if b.type == "text"), "")
    try:
        return json.loads(texto)
    except json.JSONDecodeError:
        return None


def decidir(resumen, crono):
    """Para cada narrativa, devuelve (apoya_es, titulos).

    apoya_es: {narrativa: 0 o 1}  qué bando es el que apoya
    titulos:  {narrativa: {"titulo": ..., "frases_clave": [...]}}  (solo si respondió la IA)
    """
    # Respaldo: el bando más grande es el que apoya
    apoya_es = {k: (0 if v["tamano"][0] >= v["tamano"][1] else 1) for k, v in resumen.items()}
    titulos = {}
    if not _cargar_clave():
        logging.warning("IA: no hay ANTHROPIC_API_KEY (ni en el entorno ni en .env); uso el respaldo")
        return apoya_es, titulos

    import anthropic
    cliente = anthropic.Anthropic(max_retries=3, timeout=60)
    # Hasta 8 narrativas en paralelo para no esperar una por una
    with ThreadPoolExecutor(max_workers=8) as pool:
        futuros = {k: pool.submit(_preguntar, cliente, k, v, crono) for k, v in resumen.items()}
    respondidas = 0
    for k, futuro in futuros.items():
        r = futuro.result()
        if not r:
            continue
        respondidas += 1
        apoya_es[k] = 0 if r["bando_que_afirma"] == "A" else 1
        titulos[k] = {"titulo": r["titulo"].strip(), "frases_clave": [f.strip() for f in r["frases_clave"]]}
    logging.info("IA: %s de %s narrativas resueltas con %s (US$ %.4f)",
                 respondidas, len(resumen), MODELO, crono.usd_ia)
    return apoya_es, titulos
