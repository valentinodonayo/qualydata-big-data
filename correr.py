"""El comando único del pipeline de QualyData.

Uso:
    uv run python correr.py --datos datos/practica --entrega salida/practica
    uv run python correr.py --datos datos/final --entrega entrega

Corre las etapas en orden. Regla de oro: si una etapa falla, se anota el error
en el log y se sigue con las demás. Al final SIEMPRE se escribe una entrega/
válida, aunque sea con resultados parciales.
"""
import argparse
import json
import logging
import os
import sys
import traceback
import warnings
from pathlib import Path

# Silenciamos barras de progreso y avisos de librerías: el log tiene que leerse fácil
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from pipeline import coordinacion, costo, dashboard, datos, embeddings, ia, narrativas, postura, salida, validar

RAIZ = Path(__file__).resolve().parent
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


def etapa_embeddings(estado, args):
    """Etapa 3: un vector de significado por comentario (con cache)."""
    textos = estado["datos"]["comentarios"]["texto"].tolist()
    estado["emb"], estado["emb_indice"] = embeddings.calcular(textos, args.cache)


def etapa_narrativas(estado, args):
    """Etapa 4: narrativas, ruido y fecha de origen."""
    com = estado["datos"]["comentarios"]
    # Clasificador de ruido entrenado con la verdad de práctica (si no existe, se usa una regla simple)
    ruta_modelo = args.modelo_ruido
    modelo_ruido = json.loads(ruta_modelo.read_text(encoding="utf-8")) if ruta_modelo.exists() else None
    narrativa, tabla, raiz, extra = narrativas.detectar(com, estado["emb"], estado["emb_indice"],
                                                        args.cache, modelo_ruido=modelo_ruido)
    estado["narrativa"], estado["tabla_narrativas"], estado["raiz"] = narrativa, tabla, raiz
    estado["extra_narrativas"] = extra
    estado["narrativas"] = [{"narrativa": n["narrativa"], "titulo": n["titulo"],
                             "frases_clave": n["frases_clave"],
                             "fecha_origen": salida.ts_a_iso(n["origen_ts"])} for n in tabla]


def etapa_postura(estado, args):
    """Etapa 4b: dos bandos por narrativa + IA (o respaldo) decide cuál apoya; hilos completan."""
    com = estado["datos"]["comentarios"]
    extra = estado["extra_narrativas"]
    bando_com, resumen = postura.calcular_bandos(com, estado["emb"], estado["emb_indice"], extra)
    if args.sin_ia:
        resumen_ia = ({k: (0 if v["tamano"][0] >= v["tamano"][1] else 1) for k, v in resumen.items()}, {})
        logging.info("IA: desactivada con --sin-ia; uso el bando más grande")
    else:
        resumen_ia = ia.decidir(resumen, estado["crono"])
    apoya_es, titulos = resumen_ia
    estado["postura"] = postura.posturas(com, bando_com, apoya_es, extra)
    # Títulos sin IA: el comentario más típico del bando que apoya (mejor que palabras sueltas)
    representativos = postura.titulos_representativos(com, estado["emb"], estado["emb_indice"], extra,
                                                      estado["postura"])
    for n in estado["narrativas"]:
        if n["narrativa"] in representativos:
            n["titulo"] = representativos[n["narrativa"]]
    # Si la IA puso títulos más claros, reemplazan a los provisorios
    for n in estado["narrativas"]:
        if n["narrativa"] in titulos:
            n["titulo"] = titulos[n["narrativa"]]["titulo"]
            n["frases_clave"] = "|".join(titulos[n["narrativa"]]["frases_clave"])


def etapa_coordinacion(estado, args):
    """Etapa 5: cuentas coordinadas (copias, llegadas juntas, respuestas y red de respuestas)."""
    estado["grupo"], estado["operaciones"] = coordinacion.detectar(estado["datos"]["comentarios"])
    # Guardamos el resumen de operaciones para el dashboard (bloque 7)
    ruta = args.cache / "reportes" / "operaciones.json"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(estado["operaciones"], ensure_ascii=False, indent=2), encoding="utf-8")


# Orden de las etapas. Agregar una etapa nueva = sumar una línea acá.
ETAPAS = [
    ("datos", etapa_datos),
    ("embeddings", etapa_embeddings),
    ("narrativas", etapa_narrativas),
    ("postura", etapa_postura),
    ("coordinacion", etapa_coordinacion),
]


def configurar_log(ruta_log: Path):
    """Muestra los mensajes en pantalla y además los guarda en un archivo."""
    ruta_log.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(ruta_log, mode="w", encoding="utf-8")])
    # Las librerías de descarga de modelos anotan cada pedido web: los descartamos
    # con un filtro en los "handlers" (así no importa si la librería cambia su nivel).
    ruidosas = ("httpx", "huggingface_hub", "sentence_transformers", "urllib3", "transformers")

    def filtro(registro):
        return registro.levelno >= logging.WARNING or not registro.name.startswith(ruidosas)

    for handler in logging.getLogger().handlers:
        handler.addFilter(filtro)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datos", type=Path, default=Path("datos/practica"),
                    help="Carpeta con comentarios.csv.gz, cuentas.csv.gz, publicaciones.csv.gz")
    ap.add_argument("--entrega", type=Path, default=Path("salida/practica"),
                    help="Carpeta donde se escriben los resultados")
    ap.add_argument("--cache", type=Path, default=Path("cache"),
                    help="Carpeta para la base DuckDB, Parquet y cálculos guardados")
    ap.add_argument("--modelo-ruido", type=Path, default=Path(__file__).parent / "modelos" / "ruido.json",
                    help="Coeficientes del clasificador de ruido (para experimentos de generalización)")
    ap.add_argument("--dashboard", type=Path, default=None,
                    help="Dónde escribir el dashboard HTML (por defecto reportes/dashboard.html para entrega/)")
    ap.add_argument("--sin-ia", action="store_true",
                    help="No llamar a Claude (usa el respaldo: el bando más grande apoya)")
    args = ap.parse_args()

    configurar_log(args.cache / "reportes" / "log.txt")
    crono = costo.Cronometro()
    estado = {"crono": crono}   # acá cada etapa deja sus resultados para las siguientes
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

    uso_ia = (f"IA ({ia.MODELO}) solo 1 llamada por narrativa para decidir qué bando apoya y titularla"
              if crono.tokens_ia else "IA: no se usó (sin clave); qué bando apoya lo decide el bando más grande")
    notas = (f"embeddings locales ({embeddings.MODELO}); narrativas: micro-grupos (UMAP+HDBSCAN) + votos de "
             f"hilos + Leiden; ruido: clasificador entrenado en práctica; {uso_ia}; coordinación: copias + "
             f"llegadas juntas + respuestas + comunidades densas (Leiden) + hashtag propio, sin IA")
    if errores:
        notas += f"; etapas con error: {', '.join(errores)} (ver cache/reportes/log.txt)"
    info = costo.escribir_costo(args.entrega, crono, len(estado["comentario_ids"]), ALMACENAMIENTO, notas,
                                embeddings.DISPOSITIVO_USADO)

    # Dashboard HTML (no es parte del formato oficial: si falla, la entrega ya está escrita)
    ruta_dash = args.dashboard or (RAIZ / "reportes" / "dashboard.html"
                                   if args.entrega.resolve() == (RAIZ / "entrega").resolve()
                                   else args.entrega / "dashboard.html")
    try:
        d = estado["datos"]
        datos_dash = dashboard.construir_datos(
            d["comentarios"], d["cuentas"], estado.get("narrativa") or {}, estado.get("postura") or {},
            estado.get("grupo") or {}, estado.get("narrativas") or [], estado.get("operaciones") or [],
            info, args.datos.name)
        dashboard.escribir(ruta_dash, datos_dash)
    except Exception:
        logging.error("✖ No se pudo armar el dashboard:\n%s", traceback.format_exc())

    # Validación final: ¿los 4 archivos cumplen el formato oficial?
    problemas = validar.validar_entrega(args.entrega, estado["comentario_ids"], estado["cuenta_ids"])
    if problemas:
        for p in problemas:
            logging.error("✖ Formato de entrega: %s", p)
    else:
        logging.info("✔ Formato de entrega validado (4 archivos, columnas y valores correctos)")
    logging.info("✔ Entrega en %s: %s | %.1f s", args.entrega, resumen, info["segundos"])
    if errores:
        logging.warning("Hubo errores en: %s", ", ".join(errores))
    # Código de salida 0 aunque haya errores parciales: la entrega se escribió igual
    return 0


if __name__ == "__main__":
    sys.exit(main())
