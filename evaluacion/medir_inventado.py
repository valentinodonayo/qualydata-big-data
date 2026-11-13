"""Corre el pipeline (sin IA) sobre los sets inventados y arma una tabla de resultados.

Uso:
    python evaluacion/medir_inventado.py                     # todos los sets de datos_inv/
    python evaluacion/medir_inventado.py gemelo dif_b        # solo algunos
    python evaluacion/medir_inventado.py --etiqueta arreglo  # guarda en salida/inv_arreglo/ (para comparar)

Por cada set: corre correr.py, lo puntúa con el script oficial (puntuar.py) y además mide
"bandos": qué tan bien separados quedan los dos bandos de cada narrativa SIN importar cuál
se llamó "apoya" (sin IA ese nombre es una moneda al aire cuando la narrativa está ~50/50).
"""
import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent


def bandos(carpeta_datos: Path, entrega: Path):
    """Acierto de postura si en cada narrativa real se elige la mejor orientación (apoya/rechaza)."""
    v = pd.read_csv(carpeta_datos / "verdad" / "comentarios.csv.gz", dtype=str, keep_default_na=False)
    e = pd.read_csv(entrega / "comentarios.csv", dtype=str, keep_default_na=False)
    m = v.merge(e, on="comentario_id")
    m = m[(m["narrativa_id"] != "ninguna") & (m["postura_y"] != "")]
    igual = (m["postura_x"] == m["postura_y"]).groupby(m["narrativa_id"]).mean()
    tam = m.groupby("narrativa_id").size()
    mejor = igual.where(igual >= 0.5, 1 - igual)
    return float((mejor * tam).sum() / tam.sum()), int((e["narrativa"] != "").groupby(e["narrativa"]).ngroups - (1 if (e["narrativa"] == "").any() else 0))


def medir(nombre, etiqueta, carpeta):
    datos = RAIZ / carpeta / nombre
    entrega = RAIZ / "salida" / f"inv{etiqueta}" / nombre
    subprocess.run([sys.executable, str(RAIZ / "correr.py"), "--datos", str(datos), "--entrega", str(entrega),
                    "--cache", str(RAIZ / "cache_inv" / nombre), "--sin-ia"], cwd=RAIZ, check=True,
                   capture_output=True)
    ruta_json = entrega.parent / f"{nombre}_puntaje.json"
    subprocess.run([sys.executable, str(RAIZ / "evaluacion" / "puntuar.py"), "--verdad", str(datos / "verdad"),
                    "--entrega", str(entrega), "--json", str(ruta_json)], cwd=RAIZ, check=True, capture_output=True)
    r = json.loads(ruta_json.read_text(encoding="utf-8"))
    n, c = r["narrativas"], r["coordinacion"]
    acierto_bandos, encontradas = bandos(datos, entrega)
    reales = len(pd.read_csv(datos / "verdad" / "narrativas.csv"))
    return {"set": nombre, "narr": f"{encontradas}/{reales}", "bc_P": round(n["bcubed_precision"], 3),
            "bc_R": round(n["bcubed_recall"], 3), "bc_F1": round(n["bcubed_f1"], 3),
            "ruido_F1": round(n["ruido_f1"], 3), "postura": round(n["postura_macro_f1"], 3),
            "bandos": round(acierto_bandos, 3), "origen": round(n["origen"], 3),
            "cta_P": round(c["precision"], 3), "cta_R": round(c["recall"], 3),
            "grupos": round(c["grupos_bcubed_f1"], 3), "marcadas": c["cuentas_marcadas"],
            "total": round(n["puntos"] + c["puntos"], 2), "seg": (r.get("costo") or {}).get("segundos", "")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sets", nargs="*", help="Nombres de sets (por defecto, todos los de la carpeta)")
    ap.add_argument("--carpeta", default="datos_inv")
    ap.add_argument("--etiqueta", default="", help="Sufijo de la carpeta de salida (para comparar versiones)")
    ap.add_argument("--paralelo", type=int, default=1)
    args = ap.parse_args()
    nombres = args.sets or sorted(p.name for p in (RAIZ / args.carpeta).iterdir() if (p / "verdad").exists())
    with ThreadPoolExecutor(args.paralelo) as ex:
        filas = list(ex.map(lambda s: medir(s, args.etiqueta, args.carpeta), nombres))
    tabla = pd.DataFrame(filas)
    salida = RAIZ / "salida" / f"inv{args.etiqueta}" / "tabla.csv"
    tabla.to_csv(salida, index=False)
    print(tabla.to_string(index=False))
    print(f"→ {salida.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
