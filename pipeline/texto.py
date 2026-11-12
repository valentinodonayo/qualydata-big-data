"""Normalización de texto: dejar los comentarios "parejos" antes de compararlos.

Ejemplo:  "HAY QUEEEE DECIRLO!! @majo_ok87 mirá https://x.co #ElLitioEsFuturo 💯"
      ->  "hay que decirlo!! mirá ellitioesfuturo"

Hay dos niveles:
- para_embeddings(): suave. Deja acentos y puntuación, porque el modelo los entiende.
- para_comparar():   fuerte. Saca acentos, emojis y signos. Sirve para detectar
                     textos copiados casi iguales (bots).
"""
import re
import unicodedata

_MENCION = re.compile(r"@\w+")
_LINK = re.compile(r"https?://\S+|www\.\S+")
_HASHTAG = re.compile(r"#(\w+)")
_REPETIDAS = re.compile(r"(\w)\1{2,}")          # 3 o más letras iguales seguidas
_ESPACIOS = re.compile(r"\s+")
_NO_ALFANUM = re.compile(r"[^a-zñ0-9 ]")


def _sin_emojis(texto: str) -> str:
    """Saca emojis y símbolos raros (categorías Unicode 'So', 'Sk', 'Cs')."""
    return "".join(c for c in texto if unicodedata.category(c) not in ("So", "Sk", "Cs"))


def para_embeddings(texto: str) -> str:
    """Normalización suave, para el modelo de embeddings."""
    t = (texto or "").lower()
    t = _MENCION.sub(" ", t)
    t = _LINK.sub(" ", t)
    t = _HASHTAG.sub(r"\1", t)          # "#ElLitioEsFuturo" -> "ellitioesfuturo"
    t = _REPETIDAS.sub(r"\1", t)        # "queeee" -> "que"
    t = _sin_emojis(t)
    return _ESPACIOS.sub(" ", t).strip()


def para_comparar(texto: str) -> str:
    """Normalización fuerte, para detectar textos casi idénticos."""
    t = para_embeddings(texto)
    t = t.replace("ñ", "\x00")                                     # protegemos la ñ...
    t = unicodedata.normalize("NFD", t)
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")   # saca acentos
    t = t.replace("\x00", "ñ")                                     # ...y la devolvemos
    t = _NO_ALFANUM.sub(" ", t)
    return _ESPACIOS.sub(" ", t).strip()
