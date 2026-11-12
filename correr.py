"""El comando único del pipeline de QualyData.

Uso:
    uv run python correr.py --datos datos/practica --entrega salida/practica
    uv run python correr.py --datos datos/final --entrega entrega

Corre las etapas en orden. Regla de oro: si una etapa falla, se anota el error
en el log y se sigue con las demás. Al final SIEMPRE se escribe una entrega/
válida, aunque sea con resultados parciales.
"""
import argparse
import logging
import sys
import traceback
from pathlib import Path

from pipeline import costo, datos, salida

ALMACENAMIENTO = "DuckDB local (cache/qualydata.duckdb) + copia en Parquet (cache/parquet/)"


def etapa_datos(estado, args):
    """Etapas 1 y 2: leer, limpiar y guardar."""
    estado["datos"] = datos.cargar(args.datos, args.cache)
    d = estado["datos"]
    estado["comentario_ids"] = d["comentarios"]["comentario_id"].tolist()
    estado["cuenta_ids"] = datos.todas_las_cuentas(d).tolist()
    datos.guardar_calidad(d["calidad"], args.cache / "reportes" / "calidad.json")
    logging.info("Datos: %s comentarios, %s cuentas, %s publicaciones",
                 len(d["comentarios"]), len(estado["cuenta_ids"]), len(d["publicaciones"]))


def etapa_narrativas(estado, args):
    """Etapa 4 (todavía vacía): por ahora todo queda como ruido. Llega en el bloque 3."""
    estado["narrativa"], estado["postura"], estado["narrativas"] = {}, {}, []


def etapa_coordinacion(estado, args):
    """Etapa 5 (todavía vacía): por ahora ninguna cuenta marcada. Llega en el bloque 5."""
    estado["grupo"] = {}


# Orden de las etapas. Agregar una etapa nueva = sumar una línea acá.
ETAPAS = [
    ("datos", etapa_datos),
    ("narrativas", etapa_narrativas),
    ("coordinacion", etapa_coordinacion),
]


def configurar_log(ruta_log: Path):
    """Muestra los mensajes en pantalla y además los guarda en un archivo."""
    ruta_log.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(ruta_log, mode="w", encoding="utf-8")])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datos", type=Path, default=Path("datos/practica"),
                    help="Carpeta con comentarios.csv.gz, cuentas.csv.gz, publicaciones.csv.gz")
    ap.add_argument("--entrega", type=Path, default=Path("salida/practica"),
                    help="Carpeta donde se escriben los resultados")
    ap.add_argument("--cache", type=Path, default=Path("cache"),
                    help="Carpeta para la base DuckDB, Parquet y cálculos guardados")
    args = ap.parse_args()

    configurar_log(args.cache / "reportes" / "log.txt")
    crono = costo.Cronometro()
    estado = {}       # acá cada etapa deja sus resultados para las siguientes
    errores = []      # nombres de las etapas que fallaron

    for nombre, funcion in ETAPAS:
        logging.info("▶ Etapa: %s", nombre)
        try:
            with crono.etapa(nombre):
                funcion(estado, args)
        except Exception:
            # No cortamos: anotamos el error completo y seguimos con la próxima etapa
            errores.append(nombre)
            logging.error("✖ Falló la etapa %s:\n%s", nombre, traceback.format_exc())

    # --- Escribir la entrega SIEMPRE ------------------------------------------------
    if "comentario_ids" not in estado:
        # Si ni siquiera se pudieron leer los datos, usamos el lector de respaldo
        logging.warning("Uso el lector de respaldo para obtener los ids")
        estado["comentario_ids"], estado["cuenta_ids"] = datos.ids_de_respaldo(args.datos)

    resumen = salida.escribir_entrega(
        args.entrega, estado["comentario_ids"], estado["cuenta_ids"],
        narrativa=estado.get("narrativa"), postura=estado.get("postura"),
        grupo=estado.get("grupo"), narrativas=estado.get("narrativas"))

    notas = "esqueleto del pipeline (bloque 1): sin modelos todavía"
    if errores:
        notas += f"; etapas con error: {', '.join(errores)} (ver cache/reportes/log.txt)"
    info = costo.escribir_costo(args.entrega, crono, len(estado["comentario_ids"]), ALMACENAMIENTO, notas)

    logging.info("✔ Entrega en %s: %s | %.1f s", args.entrega, resumen, info["segundos"])
    if errores:
        logging.warning("Hubo errores en: %s", ", ".join(errores))
    # Código de salida 0 aunque haya errores parciales: la entrega se escribió igual
    return 0


if __name__ == "__main__":
    sys.exit(main())
