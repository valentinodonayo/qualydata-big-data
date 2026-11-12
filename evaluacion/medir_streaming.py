"""Simula el modo "streaming" con el set de práctica: ¿alcanza con asignar al centro más cercano?

Idea para escalar a millones de comentarios por día: el clustering pesado (UMAP + HDBSCAN)
se corre cada tanto; en el medio, cada comentario nuevo se asigna a la narrativa cuyo
centro está más cerca (una búsqueda de vecino más cercano, que con un índice ANN es O(log n)).

Prueba:
  1. Centros de cada narrativa calculados SOLO con los días 1 a 7 (salida del pipeline).
  2. Las raíces de los días 8 a 14 se asignan al centro más cercano.
  3. Se compara contra la verdad: pureza (cuántos caen en la narrativa real correcta) y
     cuánto tarda por cada 1.000 comentarios.

Uso: uv run python evaluacion/medir_streaming.py   (necesita una corrida previa en salida/practica)
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import embeddings

DATOS, SALIDA, CACHE = Path("datos/practica"), Path("salida/practica"), Path("cache")

com = pd.read_csv(DATOS / "comentarios.csv.gz", dtype=str, keep_default_na=False)
pred = pd.read_csv(SALIDA / "comentarios.csv", dtype=str, keep_default_na=False)
verdad = pd.read_csv(DATOS / "verdad/comentarios.csv.gz", dtype=str, keep_default_na=False)
com = com.merge(pred, on="comentario_id").merge(verdad, on="comentario_id")
emb, indice = embeddings.calcular(com["texto"].tolist(), CACHE)
V = emb[indice]

ts = pd.to_datetime(com["fecha"], utc=True)
dia = (ts - ts.min()).dt.days
raiz = com["respuesta_a"] == ""
antes = (dia < 7).to_numpy()
despues = ((dia >= 7) & raiz & (com["narrativa"] != "")).to_numpy()

# Centros con los días 1 a 7, y la narrativa real que más aparece en cada uno (para medir)
etiquetas = sorted(set(com.loc[antes & (com["narrativa"] != ""), "narrativa"]))
centros = np.vstack([V[antes & (com["narrativa"] == e).to_numpy()].mean(0) for e in etiquetas])
centros /= np.linalg.norm(centros, axis=1, keepdims=True)
real_de = {e: com.loc[antes & (com["narrativa"] == e).to_numpy(), "narrativa_id"].value_counts().index[0]
           for e in etiquetas}

t0 = time.perf_counter()
asignada = np.array(etiquetas)[(V[despues] @ centros.T).argmax(1)]
segundos = time.perf_counter() - t0

real = com.loc[despues, "narrativa_id"].to_numpy()
acierto_streaming = np.mean([real_de[a] == r for a, r in zip(asignada, real)])
batch = com.loc[despues, "narrativa"].to_numpy()
acierto_batch = np.mean([real_de.get(b) == r for b, r in zip(batch, real)])
nuevas = sorted(set(real) - set(real_de.values()))
en_nuevas = np.isin(real, nuevas).mean()
print(f"Raíces de los días 8-14 asignadas: {despues.sum()}")
print(f"Acierto asignando al centro más cercano (streaming): {acierto_streaming:.3f}")
print(f"Acierto del pipeline completo (batch) en esas mismas raíces: {acierto_batch:.3f}")
print(f"Narrativas que nacen después del día 7 (el streaming no las conoce): {nuevas} = {en_nuevas:.1%} de esas raíces")
sin_nuevas = ~np.isin(real, nuevas)
print(f"Acierto streaming sin contar las narrativas nuevas: "
      f"{np.mean([real_de[a] == r for a, r in zip(asignada[sin_nuevas], real[sin_nuevas])]):.3f}")
print(f"Tiempo de asignación: {1000 * segundos / despues.sum() * 1000:.2f} ms por cada 1.000 comentarios")
