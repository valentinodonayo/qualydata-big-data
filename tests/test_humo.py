"""Prueba de humo de punta a punta: corre correr.py sobre una muestra chica del set de práctica,
ensuciada a propósito, y revisa que la entrega cumpla el formato.

Tarda ~1 minuto (calcula embeddings de verdad). Si no está el set de práctica, se saltea.
"""
import gzip
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from pipeline import validar

RAIZ = Path(__file__).resolve().parent.parent
PRACTICA = RAIZ / "datos" / "practica"


@pytest.mark.skipif(not (PRACTICA / "comentarios.csv.gz").exists(), reason="falta el set de práctica")
def test_pipeline_de_punta_a_punta_con_datos_sucios(tmp_path):
    com = pd.read_csv(PRACTICA / "comentarios.csv.gz", dtype=str, keep_default_na=False)
    # Muestra: todos los comentarios de 40 publicaciones (así los hilos quedan completos)
    pubs = sorted(com["publicacion_id"].unique())[::12][:40]
    muestra = com[com["publicacion_id"].isin(pubs)].copy()
    # Ensuciamos: columnas desordenadas, una extra, filas duplicadas, fechas y textos rotos
    muestra["columna_rara"] = "x"
    muestra = muestra[list(reversed(muestra.columns))]
    muestra = pd.concat([muestra, muestra.head(50)])
    muestra.iloc[:20, muestra.columns.get_loc("fecha")] = "no-es-una-fecha"
    muestra.iloc[20:40, muestra.columns.get_loc("texto")] = ""
    datos = tmp_path / "datos"
    datos.mkdir()
    with gzip.open(datos / "comentarios.csv.gz", "wt", encoding="utf-8") as f:
        muestra.to_csv(f, index=False)
    for nombre in ("cuentas.csv.gz", "publicaciones.csv.gz", "fuentes.csv"):
        (datos / nombre).write_bytes((PRACTICA / nombre).read_bytes())

    entrega = tmp_path / "entrega"
    r = subprocess.run([sys.executable, str(RAIZ / "correr.py"), "--datos", str(datos),
                        "--entrega", str(entrega), "--cache", str(tmp_path / "cache"), "--sin-ia"],
                       cwd=RAIZ, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "Falló la etapa" not in r.stdout, r.stdout[-3000:]
    ids = muestra["comentario_id"].drop_duplicates().tolist()
    assert validar.validar_entrega(entrega, ids) == []
    salida = pd.read_csv(entrega / "comentarios.csv", dtype=str, keep_default_na=False)
    assert (salida["narrativa"] != "").mean() > 0.5      # encontró narrativas en la muestra
