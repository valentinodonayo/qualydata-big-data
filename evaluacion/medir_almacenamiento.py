"""Mide cuánto tarda y cuánta memoria usa leer los comentarios con DuckDB, Polars y pandas.

Uso (Polars no es dependencia del proyecto; uv lo trae solo para esta medición):
    uv run --with polars python evaluacion/medir_almacenamiento.py duckdb
    uv run --with polars python evaluacion/medir_almacenamiento.py polars
    uv run --with polars python evaluacion/medir_almacenamiento.py pandas

Cada motor hace lo mismo: leer el .csv.gz, contar comentarios por publicación y por día,
y guardar en Parquet. Imprime segundos y memoria máxima del proceso (MB).
"""
import resource
import sys
import time
from pathlib import Path

ARCHIVO = Path("datos/practica/comentarios.csv.gz")
motor = sys.argv[1]
t0 = time.perf_counter()
if motor == "duckdb":
    import duckdb
    con = duckdb.connect()
    con.execute(f"CREATE TABLE c AS SELECT * FROM read_csv('{ARCHIVO}', all_varchar=true)")
    con.execute("SELECT publicacion_id, count(*) FROM c GROUP BY 1").fetchall()
    con.execute("SELECT substr(fecha, 1, 10) AS dia, count(*) FROM c GROUP BY 1").fetchall()
    con.execute("COPY c TO '/tmp/_medir_duckdb.parquet' (FORMAT parquet)")
elif motor == "polars":
    import polars as pl
    c = pl.read_csv(ARCHIVO, infer_schema=False)
    c.group_by("publicacion_id").len()
    c.group_by(pl.col("fecha").str.slice(0, 10)).len()
    c.write_parquet("/tmp/_medir_polars.parquet")
else:
    import pandas as pd
    c = pd.read_csv(ARCHIVO, dtype=str, keep_default_na=False)
    c.groupby("publicacion_id").size()
    c.groupby(c["fecha"].str[:10]).size()
    c.to_parquet("/tmp/_medir_pandas.parquet")
segundos = time.perf_counter() - t0
# En macOS ru_maxrss viene en bytes; en Linux, en kilobytes
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
mb = rss / 2**20 if sys.platform == "darwin" else rss / 1024
print(f"{motor},{segundos:.2f},{mb:.0f}")
