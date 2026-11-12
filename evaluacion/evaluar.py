"""Corre el pipeline + el script del jurado, y anota el resultado en experimentos/runs.csv.

Uso:
    uv run python evaluacion/evaluar.py --nombre "bloque1-esqueleto"

Así cada prueba queda registrada (fecha, commit, puntajes, segundos) y podemos
comparar con números si un cambio mejoró o empeoró las cosas.
"""
import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
REGISTRO = RAIZ / "experimentos" / "runs.csv"
COLUMNAS = ["fecha", "commit", "nombre", "total", "narrativas", "bcubed_f1", "ruido_f1",
            "postura_f1", "origen", "coordinacion", "cuentas_f1", "grupos_f1",
            "cuentas_marcadas", "segundos", "usd_ia"]


def commit_actual():
    """Devuelve el hash corto del commit. Con '*' si hay cambios sin commitear."""
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=RAIZ,
                           capture_output=True, text=True).stdout.strip()
        sucio = subprocess.run(["git", "status", "--porcelain"], cwd=RAIZ,
                               capture_output=True, text=True).stdout.strip()
        return h + ("*" if sucio else "")
    except Exception:
        return "?"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nombre", required=True, help="Nombre corto de la prueba")
    ap.add_argument("--datos", type=Path, default=RAIZ / "datos/practica")
    ap.add_argument("--entrega", type=Path, default=RAIZ / "salida/practica")
    ap.add_argument("--sin-correr", action="store_true", help="Solo puntuar una entrega ya hecha")
    args = ap.parse_args()

    # 1) Correr el pipeline con el mismo Python que está corriendo este script
    if not args.sin_correr:
        subprocess.run([sys.executable, str(RAIZ / "correr.py"), "--datos", str(args.datos),
                        "--entrega", str(args.entrega)], cwd=RAIZ, check=True)

    # 2) Correr el script oficial del jurado (sin tocarlo) y pedirle el resultado en JSON
    resultado_json = args.entrega / "_puntaje.json"
    subprocess.run([sys.executable, str(RAIZ / "evaluacion/puntuar.py"),
                    "--verdad", str(args.datos / "verdad"), "--entrega", str(args.entrega),
                    "--json", str(resultado_json)], cwd=RAIZ, check=True)
    r = json.loads(resultado_json.read_text(encoding="utf-8"))
    resultado_json.unlink()  # no lo dejamos dentro de la entrega
    n, c, k = r["narrativas"], r["coordinacion"], r.get("costo") or {}

    # 3) Anotar una fila en el registro de experimentos
    fila = {
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M"), "commit": commit_actual(),
        "nombre": args.nombre, "total": round(n["puntos"] + c["puntos"], 2),
        "narrativas": n["puntos"], "bcubed_f1": round(n["bcubed_f1"], 3),
        "ruido_f1": round(n["ruido_f1"], 3), "postura_f1": round(n["postura_macro_f1"], 3),
        "origen": round(n["origen"], 3), "coordinacion": c["puntos"], "cuentas_f1": round(c["f1"], 3),
        "grupos_f1": round(c["grupos_bcubed_f1"], 3), "cuentas_marcadas": c["cuentas_marcadas"],
        "segundos": k.get("segundos", ""), "usd_ia": k.get("usd_ia", ""),
    }
    REGISTRO.parent.mkdir(exist_ok=True)
    nuevo = not REGISTRO.exists()
    with REGISTRO.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNAS)
        if nuevo:
            w.writeheader()
        w.writerow(fila)
    print(f"\n→ Anotado en {REGISTRO.relative_to(RAIZ)}: {fila['nombre']} = {fila['total']} / 55")


if __name__ == "__main__":
    main()
