"""Etapa de IA: un modelo de lenguaje decide qué bando "apoya" cada narrativa y le pone título.

Proveedores (se usa el primero que tenga clave):
1. Vercel AI Gateway (AI_GATEWAY_API_KEY): una clave para muchos modelos. Es el que usamos.
   Votan varios modelos (MODELOS_GATEWAY) y a cada uno se le pregunta DOS veces, con los bandos
   en un orden y en el otro: si se contradice, no vota (ver _votantes y _combinar).
2. Google Gemini Flash (GEMINI_API_KEY): tiene capa gratis, sin tarjeta.
3. Claude Haiku (ANTHROPIC_API_KEY): pago, muy barato.
Sin ninguna clave: se usa la caché versionada y, si falta algo, el respaldo sin IA.

Por qué usamos IA acá y no en todo:
- Son pocas llamadas por narrativa (~13 narrativas), no una por comentario (150.000).
  Con el gateway y 3 modelos cuesta ~US$ 0,20 por corrida a precio de lista (y 0 al re-correr: caché).
- Es una decisión que necesita "entender": ¿cuál de los dos bandos sostiene
  la afirmación y cuál la niega? Ninguna regla simple acertó siempre. La IA tampoco decide sola:
  _combinar la junta con el tamaño de los bandos y con el lado de la operación coordinada.

Las claves se leen del entorno o del archivo .env, que NUNCA se sube al repositorio.

Reproducibilidad: cada respuesta se guarda en una caché versionada (modelos/cache_ia.json,
ver cache_ia.py). Con la caché, el pipeline reproduce la entrega SIN clave. Si falta una
respuesta y no hay clave (o la llamada falla), esa narrativa usa el respaldo sin IA:
"el bando más grande es el que apoya". La caché guarda huellas (hashes) de los ejemplos,
nunca los comentarios ni la clave.

Haiku 5.5 no acepta temperature (la API devuelve error con valores no estándar): la
caché es lo que hace la salida idéntica entre corridas.
"""
import hashlib
import json
import logging
import os
import threading
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

# Prompts disponibles. Cada uno entra en la huella de la caché: cambiar de prompt = preguntar de nuevo.
# "v1" es el original (lo usan Gemini y Claude directo). "v2" se midió mejor en práctica
# (ver evaluacion/medir_polaridad.py y reportes/mejoras.md).
INSTRUCCIONES_V2 = """Analizás comentarios de redes sociales en español rioplatense.
Te paso comentarios de UNA misma discusión, separados en dos bandos opuestos (A y B).

En la discusión circula una "narrativa": la afirmación ORIGINAL que alguien LANZA a la conversación.
Puede ser un reclamo, una denuncia, un rumor o una propuesta (ejemplos: "la obra está parada porque
se robaron la plata", "hay que cambiar de empresa", "la fábrica trae trabajo").
- Un bando la SOSTIENE: la dice, la repite, la refuerza con ejemplos propios.
- El otro bando REACCIONA contra ella: la desmiente, la relativiza, la minimiza, pide pruebas,
  dice que es exagerada, que es mentira o una operación, defiende al acusado o se opone a la propuesta.

Tu tarea es decir cuál bando SOSTIENE la narrativa. Cómo distinguirlos:
- La narrativa se entiende sola. La reacción solo tiene sentido como RESPUESTA a algo que otro dijo
  antes ("eso no es así", "no exageren", "otra vez con lo mismo", "dejen de repetir", "no es tan grave",
  "los datos dicen otra cosa", "así no se arregla nada").
- No te guíes por la gramática: una narrativa puede estar escrita en negativo ("no hubo consulta",
  "faltan médicos") y su desmentida en positivo ("sí hubo consulta", "médicos hay"). Tampoco por el tono:
  la narrativa puede ser una queja, un rumor alarmista o una promesa optimista.
- El bando más grande NO es necesariamente el que sostiene la narrativa.

Respondé:
- afirmacion: la narrativa en una frase, redactada tal como la diría quien la LANZA (nunca la desmentida).
- bando_que_afirma: "A" o "B", el bando que SOSTIENE esa narrativa.
- titulo: la narrativa como título corto (máximo 12 palabras), en español, afirmada como la dice ese bando.
- frases_clave: 4 a 6 palabras o frases cortas típicas de la narrativa."""

INSTRUCCIONES_V3 = """Analizás comentarios de redes sociales en español rioplatense.
Te paso comentarios de UNA misma discusión, separados en dos bandos opuestos (A y B).
El orden en que aparecen los bandos es al azar: no significa nada.

En toda discusión hay una "narrativa": lo que un bando DENUNCIA, RECLAMA, RUMOREA, PROMETE o PROPONE.
El otro bando no trae un tema propio: le CONTESTA a esa narrativa. Formas típicas de contestar:
- negarla: "eso no pasó", "sí se hizo", "es mentira", "no hay pruebas";
- minimizarla o justificarla: "no es para tanto", "era necesario", "sigue siendo barato", "en todos lados pasa";
- descalificar a quienes la dicen: "siempre los mismos criticando", "es una operación", "lo dicen los de afuera";
- bajarle el precio a una promesa o propuesta: "prometieron mucho y cumplieron poco", "eso no arregla nada",
  "tarda años", "va a salir más caro".

Para decidir, preguntate por cada bando: ¿estas frases se podrían haber dicho PRIMERO, sin que nadie
hubiera dicho nada antes? Las de la narrativa sí. Las del bando que contesta presuponen lo que dijo el otro.
No te guíes por la gramática (una narrativa puede ser "no hubo consulta" y su respuesta "sí hubo"),
ni por el tamaño de los bandos, ni por cuál aparece primero.

Respondé:
- afirmacion: la narrativa en una frase, tal como la diría quien la lanza (nunca la respuesta del otro bando).
- bando_que_afirma: "A" o "B", el bando que SOSTIENE la narrativa (el otro es el que le contesta).
- titulo: la narrativa como título corto (máximo 12 palabras), en español.
- frases_clave: 4 a 6 palabras o frases cortas típicas de la narrativa."""

PROMPTS = {"v1": INSTRUCCIONES, "v2": INSTRUCCIONES_V2, "v3": INSTRUCCIONES_V3}
PROMPT_ACTUAL = "v3"     # el que usa el proveedor "gateway"

# --- Vercel AI Gateway (AI_GATEWAY_API_KEY): una sola clave para muchos modelos, API tipo OpenAI ---
URL_GATEWAY = "https://ai-gateway.vercel.sh/v1/chat/completions"
# VOTANTES: cada modelo de esta lista da UN voto por narrativa (ver _combinar). Para cambiar de
# modelo (por ejemplo a "anthropic/claude-haiku-5.5" cuando la cuenta tenga crédito) alcanza con
# cambiar esta lista o definir la variable de entorno QUALY_MODELOS_GATEWAY="modelo1,modelo2".
MODELOS_GATEWAY = [m.strip() for m in os.environ.get(
    "QUALY_MODELOS_GATEWAY", "openai/gpt-5-mini,openai/gpt-4.1,google/gemini-2.5-flash").split(",") if m.strip()]
MODELO_GATEWAY = MODELOS_GATEWAY[0]    # el principal: sus títulos son los que se usan primero
MAX_TOKENS_GATEWAY = 4000   # holgado: los modelos que "razonan" (gpt-5) gastan tokens de salida pensando
# Precio de LISTA en US$ por token (entrada, salida), copiado de GET https://ai-gateway.vercel.sh/v1/models
# el 13/11/2026. Lo anotamos aunque la cuenta esté en la capa gratis: es lo que costaría pagarlo.
PRECIOS_GATEWAY = {
    "openai/gpt-5-mini": (0.25e-6, 2.00e-6),
    "openai/gpt-5-nano": (0.05e-6, 0.40e-6),
    "openai/gpt-5": (1.25e-6, 10.00e-6),
    "openai/gpt-4.1-mini": (0.40e-6, 1.60e-6),
    "openai/gpt-4.1": (2.00e-6, 8.00e-6),
    "mistral/mistral-small": (0.15e-6, 0.60e-6),
    "google/gemini-2.5-flash": (0.30e-6, 2.50e-6),
    "anthropic/claude-haiku-5.5": (0.10e-6, 0.50e-6),
}


def parametros_gateway(modelo):
    """Parámetros del pedido al gateway (entran en la huella de la caché)."""
    return {"proveedor": "gateway", "max_tokens": MAX_TOKENS_GATEWAY, "formato": "json_schema", "esquema": ESQUEMA}


def _post_gateway(cuerpo):
    """Un POST al gateway con la librería estándar. Devuelve (código HTTP, json o texto del error).
    La clave viaja solo en el encabezado: nunca se escribe en logs ni en la caché."""
    pedido = urllib.request.Request(
        URL_GATEWAY, data=json.dumps(cuerpo).encode("utf-8"), method="POST",
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + os.environ["AI_GATEWAY_API_KEY"]})
    try:
        with urllib.request.urlopen(pedido, timeout=120) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8", "replace")[:300]


ESPERA_REINTENTO = 20   # segundos (se multiplica por el número de intento); los tests la ponen en 0
# La capa gratis del gateway deja hacer 5 pedidos por minuto POR MODELO. Con crédito cargado se puede
# subir con la variable de entorno QUALY_PEDIDOS_POR_MINUTO (0 = sin espera).
PEDIDOS_POR_MINUTO = float(os.environ.get("QUALY_PEDIDOS_POR_MINUTO", "5"))
_candado_turnos = threading.Lock()
_turnos = {}            # por modelo: (candado, [momento del último pedido])


def _esperar_turno(modelo):
    """Hace esperar lo necesario para no pasar el límite de pedidos por minuto de ese modelo.
    Los pedidos a un mismo modelo salen de a uno, separados; modelos distintos van en paralelo."""
    if PEDIDOS_POR_MINUTO <= 0:
        return
    with _candado_turnos:
        candado, ultimo = _turnos.setdefault(modelo, (threading.Lock(), [0.0]))
    with candado:
        falta = ultimo[0] + 60.0 / PEDIDOS_POR_MINUTO + 0.5 - time.monotonic()
        if falta > 0:
            time.sleep(falta)
        ultimo[0] = time.monotonic()



def _preguntar_gateway(etiqueta, mensaje, modelo=None, instrucciones=None):
    """Una llamada a un modelo por el gateway. Devuelve la entrada para la caché, o None si no hubo
    respuesta usable (cualquier falla termina en None: quien llama sigue con los otros votos o el respaldo)."""
    modelo = modelo or MODELO_GATEWAY
    cuerpo = {
        "model": modelo,
        "max_tokens": MAX_TOKENS_GATEWAY,
        "messages": [{"role": "system", "content": instrucciones or PROMPTS[PROMPT_ACTUAL]},
                     {"role": "user", "content": mensaje}],
        # Salida estructurada: el modelo está obligado a responder un JSON con la forma de ESQUEMA
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": "narrativa", "strict": True, "schema": ESQUEMA}},
    }
    codigo, datos = None, None
    for intento in range(5):
        _esperar_turno(modelo)
        try:
            codigo, datos = _post_gateway(cuerpo)
        except Exception as error:          # sin internet, tiempo agotado, respuesta que no es JSON...
            codigo, datos = None, type(error).__name__
        if codigo == 200:
            break
        if codigo is None or codigo == 429 or codigo >= 500:   # pasajero: esperar y reintentar
            time.sleep(ESPERA_REINTENTO * (intento + 1))
            continue
        break                               # 400/401/403...: reintentar no lo arregla
    if codigo != 200:
        logging.warning("IA: falló %s con %s (HTTP %s: %s)", etiqueta, modelo, codigo, str(datos)[:160])
        return None
    try:
        respuesta = json.loads(datos["choices"][0]["message"]["content"])
        if respuesta["bando_que_afirma"] not in ("A", "B"):
            raise ValueError("bando inválido")
        respuesta = {"afirmacion": str(respuesta.get("afirmacion", "")), "bando_que_afirma": respuesta["bando_que_afirma"],
                     "titulo": str(respuesta["titulo"]), "frases_clave": [str(f) for f in respuesta["frases_clave"]]}
    except (KeyError, IndexError, TypeError, ValueError):   # incluye JSON inválido o vacío
        logging.warning("IA: %s no devolvió un JSON usable para %s", modelo, etiqueta)
        return None
    uso = datos.get("usage") or {}
    entrada, salida = int(uso.get("prompt_tokens", 0)), int(uso.get("completion_tokens", 0))
    if modelo in PRECIOS_GATEWAY:
        usd = entrada * PRECIOS_GATEWAY[modelo][0] + salida * PRECIOS_GATEWAY[modelo][1]
    else:                                   # modelo fuera de la tabla: el gateway informa el costo de lista
        usd = float(uso.get("market_cost") or uso.get("cost") or 0.0)
    return {"respuesta": respuesta, "tokens_entrada": entrada, "tokens_salida": salida, "usd": round(usd, 8)}


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
    """Devuelve el proveedor a usar según qué clave haya: "gateway", "gemini", "claude" o None."""
    if _cargar_clave("AI_GATEWAY_API_KEY"):
        return "gateway"
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


def _mensaje(info, invertido=False):
    """Arma el mensaje con los ejemplos de los dos bandos (siempre igual para los mismos datos).
    invertido=True presenta los bandos al revés (el bando 1 como "A"): sirve para detectar si el
    modelo elige "A" solo por estar primero."""
    bloques = []
    # "variados" = una frase por cada forma distinta de decirlo (ver postura._ejemplos_variados)
    listas = info.get("variados") or info["ejemplos"]
    for nombre, ejemplos in zip("AB", listas[::-1] if invertido else listas):
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


def _huella_prompt(instrucciones):
    """Huella corta del prompt: una respuesta guardada solo se reutiliza para el MISMO prompt."""
    return hashlib.sha256(instrucciones.encode("utf-8")).hexdigest()[:12]


def _huellas_ejemplos(ejemplos):
    """Cambia cada comentario de ejemplo por una huella corta (hash). En la caché, que es pública,
    se guardan SOLO estas huellas: alcanzan para reconocer un bando por sus ejemplos en común,
    sin copiar al repositorio comentarios del set de datos (el set final no se puede subir)."""
    return [[hashlib.sha256(t.encode("utf-8")).hexdigest()[:16] for t in lista] for lista in ejemplos]


def _buscar_parecida(cache, huella_datos, ejemplos, modelo=None, voto=0, prompt=None):
    """Si no hay coincidencia exacta: busca una respuesta guardada del MISMO set de datos (y mismo
    modelo, voto y prompt) y reconoce cada bando por sus ejemplos en común (los ejemplos son los más
    centrales, así que casi no cambian entre máquinas ni con retoques chicos del pipeline).
    No exige la misma etiqueta: si las narrativas cambian de número (n07 pasa a ser n08), los
    ejemplos igual la identifican. Devuelve (entrada, invertida) o (None, False).
    """
    mejor, invertida, mejor_puntaje = None, False, 0
    ejemplos = _huellas_ejemplos(ejemplos)      # en la caché están guardadas las huellas, no los textos
    for clave in sorted(cache.datos):          # orden fijo: mismo resultado en cualquier máquina
        entrada = cache.datos[clave]
        if entrada.get("huella_datos") != huella_datos:
            continue
        if entrada.get("modelo") not in (None, modelo) or entrada.get("voto", 0) != voto \
                or entrada.get("huella_prompt") not in (None, prompt):
            continue
        a, b = (set(x) for x in entrada.get("ejemplos", [[], []]))
        directo = len(a & set(ejemplos[0])) + len(b & set(ejemplos[1]))
        cruzado = len(a & set(ejemplos[1])) + len(b & set(ejemplos[0]))
        if max(directo, cruzado) >= 2 * MIN_COINCIDENCIAS and max(directo, cruzado) > mejor_puntaje \
                and directo != cruzado:
            mejor, invertida, mejor_puntaje = entrada, cruzado > directo, max(directo, cruzado)
    return mejor, invertida


# --- Regla final: cómo se combinan los votos de la IA con las señales sin IA ---------------------
MARGEN_GRANDE = 0.54    # si un bando tiene ≥ 54% de la narrativa, el tamaño es una señal fuerte
MIN_OPERACION = 0.80    # la operación coordinada cuenta si ≥ 80% de sus comentarios cae en un bando
PUNTOS_OPERACION = 2.0  # cuánto pesa cada señal sin IA (cada modelo que vota pesa 1 punto)
PUNTOS_MARGEN_GRANDE = 1.5
PUNTOS_MARGEN_CHICO = 0.5


def _combinar(votos, tamano, coordinadas=None):
    """Decide qué bando apoya (0 o 1) juntando tres señales. Devuelve (bando, motivo).

    votos:       lista de 0/1, un voto por cada modelo que respondió sin contradecirse (puede estar vacía).
    tamano:      [comentarios del bando 0, del bando 1].
    coordinadas: [n0, n1] comentarios de la operación coordinada en cada bando, o None.

    Idea: un error en UNA narrativa cuesta ~0,5 puntos, así que ninguna señal débil decide sola.
    1. Si votaron 2 o más modelos y TODOS coinciden: gana la IA (en práctica, 0 errores así).
    2. Si no (un solo voto, votos divididos o ninguno), se suman puntos por bando:
       - 1 punto por cada modelo que lo votó;
       - 2 puntos si la operación coordinada escribe en ese bando (en práctica, las 3 campañas
         escriben a favor de la narrativa que empujan);
       - al bando más grande: 1,5 puntos si el margen es grande (≥ 54%: en práctica el bando que
         apoya nunca bajó de 49,5%, así que un bando tan grande casi seguro es el que apoya),
         y si no 0,5 (solo desempata).
       Gana el que junta más puntos. Sin IA y sin operación esto es la regla de siempre:
       "el bando más grande apoya".
    """
    mayor = 0 if tamano[0] >= tamano[1] else 1
    margen_grande = max(tamano) / max(tamano[0] + tamano[1], 1) >= MARGEN_GRANDE
    operacion = None
    if coordinadas and sum(coordinadas) > 0 and max(coordinadas) / sum(coordinadas) >= MIN_OPERACION:
        operacion = 0 if coordinadas[0] > coordinadas[1] else 1

    if len(votos) >= 2 and len(set(votos)) == 1:
        return votos[0], f"IA unánime ({len(votos)} modelos)"
    puntos = [0.0, 0.0]
    partes = []
    for v in votos:
        puntos[v] += 1
    if votos:
        partes.append(f"IA {votos.count(0)}-{votos.count(1)}")
    if operacion is not None:
        puntos[operacion] += PUNTOS_OPERACION
        partes.append(f"operación coordinada en {'AB'[operacion]}")
    puntos[mayor] += PUNTOS_MARGEN_GRANDE if margen_grande else PUNTOS_MARGEN_CHICO
    partes.append(f"bando más grande {'AB'[mayor]}" + (" con margen grande" if margen_grande else ""))
    ganador = mayor if puntos[0] == puntos[1] else (0 if puntos[0] > puntos[1] else 1)
    return ganador, ("sin IA: " if not votos else "por puntos: ") + ", ".join(partes)


def _votantes(proveedor):
    """Lista de pedidos que se hacen por cada narrativa. Cada uno es un dict con:
    modelo, parametros, instrucciones, voto (número de pedido) e invertido (bandos presentados al revés).

    Gemini y Claude directo hacen un solo pedido. El gateway hace DOS por cada modelo de
    MODELOS_GATEWAY: uno con los bandos en orden y otro al revés. Medimos en práctica que los modelos
    tienden a elegir "A" por estar primero; preguntando de las dos formas, ese sesgo se nota
    (el modelo se contradice) y ese modelo no vota en esa narrativa."""
    if proveedor != "gateway":
        modelo, parametros = _config(proveedor)
        return [{"modelo": modelo, "parametros": parametros, "instrucciones": INSTRUCCIONES,
                 "voto": 0, "invertido": False}]
    lista = []
    for modelo in dict.fromkeys(MODELOS_GATEWAY):       # sin repetidos, en el orden de la lista
        for invertido in (False, True):
            lista.append({"modelo": modelo, "parametros": {**parametros_gateway(modelo), "invertido": invertido},
                          "instrucciones": PROMPTS[PROMPT_ACTUAL], "voto": len(lista), "invertido": invertido})
    return lista


def _preguntar_votante(votante, proveedor, etiqueta, mensaje, cliente=None):
    """Hace el pedido de un votante con el proveedor que corresponda."""
    if proveedor == "gateway":
        return _preguntar_gateway(etiqueta, mensaje, votante["modelo"], votante["instrucciones"])
    if proveedor == "gemini":
        return _preguntar_gemini(etiqueta, mensaje)
    return _preguntar(cliente, etiqueta, mensaje)


def votos_por_modelo(respuestas):
    """Convierte las respuestas de una narrativa en votos, uno por modelo como máximo.

    respuestas: lista de (modelo, bando elegido 0/1) — hasta dos por modelo (orden normal e invertido).
    Si un modelo respondió dos veces y se contradice, NO vota (eligió por la posición, no por el
    contenido). Devuelve (votos, contradicciones): lista de 0/1 y cuántos modelos se contradijeron."""
    por_modelo = {}
    for modelo, bando in respuestas:
        por_modelo.setdefault(modelo, []).append(bando)
    votos, contradicciones = [], 0
    for bandos in por_modelo.values():
        if len(set(bandos)) == 1:
            votos.append(bandos[0])
        else:
            contradicciones += 1
    return votos, contradicciones


def decidir(resumen, crono, ruta_cache=None, huella_datos=""):
    """Para cada narrativa, devuelve (apoya_es, titulos).

    apoya_es: {narrativa: 0 o 1}  qué bando es el que apoya
    titulos:  {narrativa: {"titulo": ..., "frases_clave": [...]}}  (solo si hubo respuesta de la IA)

    Orden por cada pedido: 1) caché versionada, 2) modelo (si hay clave), 3) sin respuesta.
    Con las respuestas que haya, _combinar decide (sin ninguna = respaldo sin IA).
    El costo que se suma a costo.json es el de la corrida que generó cada respuesta.
    """
    cache = cache_ia.CacheIA(ruta_cache) if ruta_cache else cache_ia.CacheIA()
    # Proveedor: el que tenga clave. Sin clave se usa la huella del proveedor que llenó la caché
    # (así el jurado reproduce la entrega): gateway, si no Gemini, y si no Claude.
    proveedor = _elegir_proveedor()
    hay_clave = proveedor is not None
    if proveedor is None:
        if any(e.get("proveedor") == "gateway" for e in cache.datos.values()):
            proveedor = "gateway"
        elif any(e.get("modelo") == MODELO_GEMINI for e in cache.datos.values()):
            proveedor = "gemini"
        else:
            proveedor = "claude"
    votantes = _votantes(proveedor)

    def ejemplos_de(k, votante):
        """Ejemplos de la narrativa en el orden en que ese votante los ve (para reconocerla en la caché)."""
        return resumen[k]["ejemplos"][::-1] if votante["invertido"] else resumen[k]["ejemplos"]

    # 1) Caché: coincidencia exacta y, si no, una respuesta parecida (mismos ejemplos)
    mensajes, claves, entradas, invertidas = {}, {}, {}, set()
    for k in sorted(resumen):
        for v in votantes:
            kv = (k, v["voto"])
            mensajes[kv] = _mensaje(resumen[k], v["invertido"])
            claves[kv] = cache_ia.huella(v["modelo"], v["parametros"], v["instrucciones"], mensajes[kv])
            entradas[kv] = cache.buscar(claves[kv])
            if entradas[kv] is None:
                parecida, invertida = _buscar_parecida(cache, huella_datos, ejemplos_de(k, v), v["modelo"],
                                                       v["voto"], _huella_prompt(v["instrucciones"]))
                if parecida:
                    entradas[kv] = parecida
                    cache.aciertos += 1
                    if invertida:
                        invertidas.add(kv)
                    logging.info("IA: %s (%s) resuelta con una respuesta guardada parecida (ejemplos en común)",
                                 k, v["modelo"])
    faltan = sorted(kv for kv, e in entradas.items() if e is None)

    # 2) Modelo: solo lo que falta y solo si hay clave
    if faltan and hay_clave:
        por_voto = {v["voto"]: v for v in votantes}
        if proveedor == "gemini":
            # De a una: la capa gratis limita los pedidos por minuto
            resultados = {(k, n): _preguntar_votante(por_voto[n], proveedor, k, mensajes[k, n]) for k, n in faltan}
        else:
            cliente = None
            if proveedor == "claude":
                import anthropic
                cliente = anthropic.Anthropic(max_retries=3, timeout=60)
            if proveedor == "gateway" and PEDIDOS_POR_MINUTO > 0:
                minutos = len(faltan) / len(set(v["modelo"] for v in votantes)) / PEDIDOS_POR_MINUTO
                logging.info("IA: %s pedidos al gateway; con el límite de %g por minuto por modelo tarda ~%.0f min "
                             "(una sola vez: después se usa la caché)", len(faltan), PEDIDOS_POR_MINUTO, minutos)
            # En paralelo. En el gateway cada modelo respeta su propio límite por minuto (_esperar_turno)
            with ThreadPoolExecutor(max_workers=16) as pool:
                futuros = {(k, n): pool.submit(_preguntar_votante, por_voto[n], proveedor, k, mensajes[k, n], cliente)
                           for k, n in faltan}
            resultados = {kv: f.result() for kv, f in futuros.items()}
        for (k, n), entrada in sorted(resultados.items()):
            if entrada:
                v = por_voto[n]
                entrada.update({"etiqueta": k, "huella_datos": huella_datos,
                                "ejemplos": _huellas_ejemplos(ejemplos_de(k, v)),
                                "modelo": v["modelo"], "voto": n, "proveedor": proveedor,
                                "huella_prompt": _huella_prompt(v["instrucciones"])})
                cache.guardar(claves[k, n], entrada)
                entradas[k, n] = entrada
        cache.escribir()
    elif faltan:
        logging.warning("IA: %s respuestas que no están en la caché y no hay clave (AI_GATEWAY_API_KEY, "
                        "GEMINI_API_KEY ni ANTHROPIC_API_KEY): esas narrativas se deciden con lo que haya "
                        "(sin ningún voto de IA = el bando más grande apoya)", len(faltan))

    # 3) Combinar los votos de cada narrativa con las señales sin IA
    apoya_es, titulos, detalle = {}, {}, {}
    for k in sorted(resumen):
        respuestas = []          # (modelo, bando elegido, respuesta completa)
        for v in votantes:
            e = entradas[k, v["voto"]]
            if not e:
                continue
            crono.sumar_ia(e["tokens_entrada"] + e["tokens_salida"], e["usd"])
            bando = 0 if e["respuesta"]["bando_que_afirma"] == "A" else 1
            if v["invertido"]:                    # a este votante se le mostró el bando 1 como "A"
                bando = 1 - bando
            if (k, v["voto"]) in invertidas:      # la respuesta guardada tenía los bandos al revés
                bando = 1 - bando
            respuestas.append((v["modelo"], bando, e["respuesta"]))
        votos, contradicciones = votos_por_modelo([(m, b) for m, b, _ in respuestas])
        apoya_es[k], motivo = _combinar(votos, resumen[k]["tamano"], resumen[k].get("coordinadas"))
        if respuestas:
            # Título: el de la primera respuesta que coincide con la decisión final (si ninguna, la primera)
            r = next((r for _, b, r in respuestas if b == apoya_es[k]), respuestas[0][2])
            titulos[k] = {"titulo": r["titulo"].strip(), "frases_clave": [f.strip() for f in r["frases_clave"]]}
        mayor = 0 if resumen[k]["tamano"][0] >= resumen[k]["tamano"][1] else 1
        detalle[k] = {"apoya": apoya_es[k], "motivo": motivo, "votos": votos, "contradicciones": contradicciones,
                      "mayoria": mayor, "tamano": resumen[k]["tamano"],
                      "coordinadas": resumen[k].get("coordinadas")}
        logging.info("IA: %s -> apoya el bando %s (%s; votos %s, %s modelos se contradicen, tamaños %s, operación %s)",
                     k, "AB"[apoya_es[k]], motivo, "".join("AB"[v] for v in votos) or "-", contradicciones,
                     resumen[k]["tamano"], resumen[k].get("coordinadas"))
    nombres = " + ".join(dict.fromkeys(v["modelo"] for v in votantes))
    crono.ia_desde_cache = cache.aciertos
    crono.ia_nuevas = cache.nuevas
    crono.ia_modelo = nombres   # para que costo.json diga qué modelo(s) se usaron
    crono.ia_pedidos = len(votantes)        # pedidos por narrativa
    crono.ia_detalle = detalle              # para el log / reportes: por qué se decidió cada narrativa
    logging.info("IA: %s de %s narrativas con respuesta de %s (%s respuestas de la caché, %s nuevas; US$ %.4f)",
                 len(titulos), len(resumen), nombres, cache.aciertos, cache.nuevas, crono.usd_ia)
    return apoya_es, titulos
