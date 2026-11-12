"""Etapa 3: convertir cada comentario en un vector ("embedding").

Un embedding es una lista de 384 números que representa el SIGNIFICADO del texto.
Dos comentarios que dicen lo mismo con distintas palabras quedan con vectores
parecidos ("los bondis se caen a pedazos" ≈ "el servicio de colectivos es pésimo").

Modelo: intfloat/multilingual-e5-small
- Multilingüe y entrenado para similitud de frases. Corre local, sin pagar nada.
- Licencia MIT, no pide cuenta ni clave.

Trucos para que sea rápido y barato:
1. Cada texto DISTINTO se embeddea una sola vez (los repetidos se reutilizan).
2. Se usa la GPU de la Mac (MPS) si está disponible.
3. Los vectores se guardan en cache/: si se vuelve a correr con los mismos textos,
   no se recalcula nada.
"""
import hashlib
import logging
import os
from pathlib import Path

import numpy as np

from pipeline import texto as tx

MODELO = "intfloat/multilingual-e5-small"
# Versión exacta del modelo en Hugging Face: si el autor lo actualiza, seguimos usando esta
REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
PREFIJO = "query: "        # los modelos e5 esperan este prefijo delante de cada texto
TAMANO_LOTE = 256           # cuántos textos procesa juntos
LARGO_MAXIMO = 64           # los comentarios son cortos: cortar en 64 tokens acelera mucho


def _elegir_dispositivo():
    """Usa la GPU de la Mac (MPS) si existe; si no, el procesador (CPU).

    Se puede forzar con la variable de entorno QUALY_DISPOSITIVO=cpu (o mps / cuda).
    """
    import torch
    forzado = os.environ.get("QUALY_DISPOSITIVO", "").strip().lower()
    if forzado:
        return forzado
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def calcular(textos, carpeta_cache: Path):
    """Devuelve (matriz, indice).

    - matriz: un vector por cada texto DISTINTO (filas normalizadas, largo 1).
    - indice: para cada comentario, qué fila de la matriz le corresponde.
      Así el comentario i tiene su vector en matriz[indice[i]].
    """
    normalizados = [tx.para_embeddings(t) for t in textos]
    # Textos únicos (ordenados alfabéticamente) y a cuál de ellos corresponde cada comentario
    unicos, indice = np.unique(np.array(normalizados, dtype=object), return_inverse=True)
    unicos = unicos.tolist()
    logging.info("Embeddings: %s comentarios -> %s textos distintos", len(textos), len(unicos))

    # La "huella" (hash) depende del modelo y de los textos: si algo cambia, se recalcula
    huella = hashlib.sha1((MODELO + "\n" + "\n".join(unicos)).encode("utf-8")).hexdigest()[:16]
    ruta = carpeta_cache / "embeddings" / f"{huella}.npy"
    if ruta.exists():
        logging.info("Embeddings: uso la cache %s", ruta.name)
        return np.load(ruta), indice

    from sentence_transformers import SentenceTransformer   # se importa acá porque tarda
    dispositivo = _elegir_dispositivo()
    logging.info("Embeddings: calculando con %s en %s", MODELO, dispositivo)
    modelo = SentenceTransformer(MODELO, device=dispositivo, revision=REVISION)
    modelo.max_seq_length = LARGO_MAXIMO
    # Un texto vacío no tiene significado: lo reemplazamos por un punto para no romper el modelo
    entrada = [PREFIJO + (t if t else ".") for t in unicos]
    matriz = modelo.encode(entrada, batch_size=TAMANO_LOTE, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=False).astype(np.float32)

    ruta.parent.mkdir(parents=True, exist_ok=True)
    np.save(ruta, matriz)
    return matriz, indice
