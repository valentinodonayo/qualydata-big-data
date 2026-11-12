"""Etapa 1 y 2: leer, limpiar y guardar los datos.

Por qué DuckDB:
- Es una base de datos que vive en UN archivo (cache/qualydata.duckdb), sin servidor.
  El jurado la puede re-correr con un solo comando, sin instalar nada aparte.
- Es "columnar": está hecha para hacer cuentas sobre millones de filas
  (ej: cuántos comentarios por narrativa por día) mucho más rápido que SQLite.
- Lee los .csv.gz directo y puede APARTAR las filas rotas en vez de cortar todo.

Además guardamos una copia limpia en Parquet (formato estándar, comprimido),
que se puede abrir desde cualquier herramienta (pandas, Spark, BigQuery...).
"""
import csv
import gzip
import json
from pathlib import Path

import duckdb
import pandas as pd

# Columnas que esperamos en cada archivo. Las leemos TODAS como texto (VARCHAR)
# y las convertimos después con TRY_CAST: si un valor está roto, queda vacío
# (NULL) en lugar de hacer caer el programa.
ESQUEMAS = {
    "comentarios": ["comentario_id", "publicacion_id", "cuenta_id", "fecha", "texto",
                    "respuesta_a", "me_gusta"],
    "cuentas": ["cuenta_id", "usuario", "fecha_creacion", "seguidores", "siguiendo"],
    "publicaciones": ["publicacion_id", "fuente_id", "fecha", "titular"],
    "fuentes": ["fuente_id", "nombre", "tipo"],
}


def _buscar_archivo(carpeta: Path, nombre: str):
    """Devuelve la ruta a nombre.csv.gz o nombre.csv (lo que exista), o None."""
    for opcion in (carpeta / f"{nombre}.csv.gz", carpeta / f"{nombre}.csv"):
        if opcion.exists():
            return opcion
    return None


def _leer_crudo(con, nombre: str, ruta, calidad: dict):
    """Lee un CSV a una tabla 'crudo_<nombre>' con todas las columnas como texto.

    - store_rejects: las filas que no se pueden leer van a una tabla aparte y se cuentan.
    - strict_mode=false: tolera comillas mal cerradas y filas con columnas de más.
    - null_padding: si a una fila le faltan columnas, se completan con vacío.
    - Si al archivo le falta una columna entera, la agregamos vacía.
    """
    columnas = ESQUEMAS[nombre]
    tabla = f"crudo_{nombre}"
    if ruta is None:
        # El archivo no existe: creamos una tabla vacía con las columnas esperadas
        definicion = ", ".join(f'"{c}" VARCHAR' for c in columnas)
        con.execute(f"CREATE OR REPLACE TABLE {tabla} ({definicion})")
        calidad[f"{nombre}_archivo_faltante"] = True
        return
    con.execute(f"""
        CREATE OR REPLACE TABLE {tabla} AS
        SELECT * FROM read_csv(?, header=true, all_varchar=true, store_rejects=true,
                               strict_mode=false, null_padding=true,
                               rejects_table='rechazos_{nombre}',
                               rejects_scan='escaneos_{nombre}')
    """, [str(ruta)])
    calidad[f"{nombre}_filas_rechazadas"] = con.execute(
        f"SELECT count(*) FROM rechazos_{nombre}").fetchone()[0]
    # Columnas que faltan en el archivo -> las agregamos vacías para que lo demás no falle
    existentes = {fila[0] for fila in con.execute(f"DESCRIBE {tabla}").fetchall()}
    faltantes = [c for c in columnas if c not in existentes]
    for c in faltantes:
        con.execute(f'ALTER TABLE {tabla} ADD COLUMN "{c}" VARCHAR')
    if faltantes:
        calidad[f"{nombre}_columnas_faltantes"] = faltantes


def cargar(carpeta_datos: Path, carpeta_cache: Path):
    """Lee los 4 archivos, los limpia, los guarda en DuckDB + Parquet y los devuelve.

    Devuelve un diccionario con tres DataFrames de pandas:
      "comentarios": comentario_id, publicacion_id, cuenta_id, ts, fecha, texto, respuesta_a, me_gusta
      "cuentas":     cuenta_id, usuario, fecha_creacion, seguidores, siguiendo
      "publicaciones": publicacion_id, fuente_id, ts, titular
    y "calidad": un resumen de los problemas encontrados en los datos.

    'ts' es la fecha como número (segundos desde 1970, "epoch"). Es más simple
    para hacer cuentas de tiempo que lidiar con zonas horarias.
    """
    carpeta_cache.mkdir(parents=True, exist_ok=True)
    calidad = {}
    # Archivo nuevo en cada corrida: así no se mezclan datos de práctica con los del final
    ruta_db = carpeta_cache / "qualydata.duckdb"
    if ruta_db.exists():
        ruta_db.unlink()
    con = duckdb.connect(str(ruta_db))

    for nombre in ESQUEMAS:
        _leer_crudo(con, nombre, _buscar_archivo(carpeta_datos, nombre), calidad)

    # --- Limpieza de comentarios -------------------------------------------------
    # 1) Sacamos espacios de más en los ids y descartamos filas sin id.
    # 2) Si un comentario_id aparece repetido, nos quedamos con el primero.
    # 3) La fecha se convierte con TRY_CAST: si está rota queda NULL (y la contamos).
    con.execute("""
        CREATE OR REPLACE TABLE comentarios AS
        WITH base AS (
            SELECT trim(comentario_id) AS comentario_id,
                   trim(publicacion_id) AS publicacion_id,
                   coalesce(nullif(trim(cuenta_id), ''), 'desconocida') AS cuenta_id,
                   trim(fecha) AS fecha,
                   coalesce(texto, '') AS texto,
                   nullif(trim(respuesta_a), '') AS respuesta_a,
                   coalesce(TRY_CAST(me_gusta AS BIGINT), 0) AS me_gusta,
                   row_number() OVER () AS orden
            FROM crudo_comentarios
            WHERE comentario_id IS NOT NULL AND trim(comentario_id) <> ''
        ),
        sin_duplicados AS (
            SELECT * FROM base
            QUALIFY row_number() OVER (PARTITION BY comentario_id ORDER BY orden) = 1
        )
        SELECT comentario_id, publicacion_id, cuenta_id,
               CAST(epoch(TRY_CAST(fecha AS TIMESTAMPTZ)) AS BIGINT) AS ts,
               fecha, texto, respuesta_a, me_gusta
        FROM sin_duplicados
        ORDER BY orden
    """)
    crudos = con.execute("SELECT count(*) FROM crudo_comentarios").fetchone()[0]
    limpios = con.execute("SELECT count(*) FROM comentarios").fetchone()[0]
    calidad["comentarios_leidos"] = crudos
    calidad["comentarios_limpios"] = limpios
    calidad["comentarios_descartados_sin_id_o_duplicados"] = crudos - limpios
    calidad["comentarios_fecha_invalida"] = con.execute(
        "SELECT count(*) FROM comentarios WHERE ts IS NULL").fetchone()[0]
    calidad["comentarios_texto_vacio"] = con.execute(
        "SELECT count(*) FROM comentarios WHERE trim(texto) = ''").fetchone()[0]
    calidad["respuestas_a_comentario_inexistente"] = con.execute("""
        SELECT count(*) FROM comentarios c
        WHERE c.respuesta_a IS NOT NULL
          AND c.respuesta_a NOT IN (SELECT comentario_id FROM comentarios)
    """).fetchone()[0]

    # --- Limpieza de cuentas y publicaciones ---------------------------------------
    con.execute("""
        CREATE OR REPLACE TABLE cuentas AS
        WITH base AS (SELECT *, row_number() OVER () AS orden FROM crudo_cuentas)
        SELECT trim(cuenta_id) AS cuenta_id, usuario,
               TRY_CAST(fecha_creacion AS DATE) AS fecha_creacion,
               TRY_CAST(seguidores AS BIGINT) AS seguidores,
               TRY_CAST(siguiendo AS BIGINT) AS siguiendo
        FROM base
        WHERE cuenta_id IS NOT NULL AND trim(cuenta_id) <> ''
        -- Si una cuenta aparece repetida, gana la PRIMERA fila del archivo (siempre la misma)
        QUALIFY row_number() OVER (PARTITION BY trim(cuenta_id) ORDER BY orden) = 1
        -- ORDER BY fijo: DuckDB trabaja en paralelo y sin esto el orden cambia entre corridas
        ORDER BY orden
    """)
    con.execute("""
        CREATE OR REPLACE TABLE publicaciones AS
        WITH base AS (SELECT *, row_number() OVER () AS orden FROM crudo_publicaciones)
        SELECT trim(publicacion_id) AS publicacion_id, trim(fuente_id) AS fuente_id,
               CAST(epoch(TRY_CAST(fecha AS TIMESTAMPTZ)) AS BIGINT) AS ts,
               coalesce(titular, '') AS titular
        FROM base
        WHERE publicacion_id IS NOT NULL AND trim(publicacion_id) <> ''
        QUALIFY row_number() OVER (PARTITION BY trim(publicacion_id) ORDER BY orden) = 1
        ORDER BY orden
    """)
    calidad["cuentas"] = con.execute("SELECT count(*) FROM cuentas").fetchone()[0]
    calidad["publicaciones"] = con.execute("SELECT count(*) FROM publicaciones").fetchone()[0]

    # --- Copia en Parquet (formato estándar y comprimido) ---------------------------
    carpeta_parquet = carpeta_cache / "parquet"
    carpeta_parquet.mkdir(exist_ok=True)
    for tabla in ("comentarios", "cuentas", "publicaciones"):
        con.execute(f"COPY {tabla} TO '{carpeta_parquet / tabla}.parquet' (FORMAT parquet)")

    resultado = {
        "comentarios": con.execute("SELECT * FROM comentarios").df(),
        "cuentas": con.execute("SELECT * FROM cuentas").df(),
        "publicaciones": con.execute("SELECT * FROM publicaciones").df(),
        "calidad": calidad,
    }
    con.close()
    return resultado


def ids_de_respaldo(carpeta_datos: Path):
    """Plan B si DuckDB falla: lee solo los ids con la librería estándar de Python.

    Sirve para escribir una entrega válida (aunque vacía de resultados) pase lo que pase.
    Devuelve (lista de comentario_id, lista de cuenta_id).
    """
    comentarios, cuentas = [], set()
    ruta = _buscar_archivo(carpeta_datos, "comentarios")
    if ruta:
        abrir = gzip.open if ruta.suffix == ".gz" else open
        with abrir(ruta, "rt", encoding="utf-8", errors="replace", newline="") as f:
            for fila in csv.DictReader(f):
                if fila.get("comentario_id"):
                    comentarios.append(fila["comentario_id"].strip())
                if fila.get("cuenta_id"):
                    cuentas.add(fila["cuenta_id"].strip())
    return list(dict.fromkeys(comentarios)), sorted(cuentas)


def guardar_calidad(calidad: dict, ruta: Path):
    """Guarda el resumen de calidad de datos como JSON legible."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(calidad, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def todas_las_cuentas(datos: dict) -> pd.Series:
    """Cuentas del archivo de cuentas + las que aparecen comentando (sin repetir)."""
    ids = pd.concat([datos["cuentas"]["cuenta_id"], datos["comentarios"]["cuenta_id"]])
    return ids[ids != "desconocida"].dropna().drop_duplicates().reset_index(drop=True)
