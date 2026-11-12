"""¿El pipeline generaliza a temas que nunca vio? Simula el set final con el set de práctica.

Uso:
    uv run python evaluacion/generalizacion.py

Experimentos (todos sin IA, puntuados con el script oficial del jurado):

1. TEMA AFUERA: para cada tema de práctica (litio, colectivo, hospital):
   - se entrena el clasificador de ruido SOLO con los otros dos temas,
   - se corre el pipeline SOLO sobre las publicaciones del tema que quedó afuera.
   Es lo más parecido al set final: un tema nuevo que el pipeline nunca vio.
2. CORTE TEMPORAL: solo los primeros 10 días (el set final dura 10).
3. SEMILLAS: el set completo con semillas 1, 2 y 3 (¿cambia mucho el resultado?).
4. DATOS SUCIOS: el set completo con columnas desordenadas, filas duplicadas,
   fechas rotas, textos vacíos y cuentas sin metadata.

Escribe una tabla en docs/generalizacion.md y los números en experimentos/generalizacion.csv.
"""
import csv
import gzip
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
PRACTICA = RAIZ / "datos" / "practica"
TRABAJO = RAIZ / "salida" / "generalizacion"
SEMILLA_DATOS = 7   # semilla para ensuciar los datos (fija: el experimento se puede repetir)


def _leer(nombre, carpeta=PRACTICA):
    return pd.read_csv(carpeta / nombre, dtype=str, keep_default_na=False)


def _guardar_set(carpeta: Path, com, cue, pub, verdad_com, verdad_cue, verdad_narr):
    """Escribe un set de datos (con su carpeta verdad/) en el mismo formato que el oficial."""
    (carpeta / "verdad").mkdir(parents=True, exist_ok=True)
    for df, nombre in ((com, "comentarios.csv.gz"), (cue, "cuentas.csv.gz"), (pub, "publicaciones.csv.gz"),
                       (verdad_com, "verdad/comentarios.csv.gz"), (verdad_cue, "verdad/cuentas.csv.gz")):
        with gzip.open(carpeta / nombre, "wt", encoding="utf-8") as f:
            df.to_csv(f, index=False)
    verdad_narr.to_csv(carpeta / "verdad" / "narrativas.csv", index=False)
    (carpeta / "fuentes.csv").write_bytes((PRACTICA / "fuentes.csv").read_bytes())


def _subconjunto(carpeta: Path, mascara_com):
    """Guarda el set de práctica restringido a los comentarios de la máscara."""
    com, cue, pub = _leer("comentarios.csv.gz"), _leer("cuentas.csv.gz"), _leer("publicaciones.csv.gz")
    v_com, v_cue = _leer("verdad/comentarios.csv.gz"), _leer("verdad/cuentas.csv.gz")
    v_narr = _leer("verdad/narrativas.csv")
    sub = com[mascara_com]
    cuentas = set(sub["cuenta_id"])
    narrs = set(v_com[v_com["comentario_id"].isin(sub["comentario_id"])]["narrativa_id"])
    _guardar_set(carpeta, sub, cue[cue["cuenta_id"].isin(cuentas)],
                 pub[pub["publicacion_id"].isin(set(sub["publicacion_id"]))],
                 v_com[v_com["comentario_id"].isin(sub["comentario_id"])],
                 v_cue[v_cue["cuenta_id"].isin(cuentas)], v_narr[v_narr["narrativa_id"].isin(narrs)])


def _tema_por_publicacion():
    """Tema de cada publicación = el tema de la mayoría de sus comentarios con narrativa (de la verdad)."""
    com = _leer("comentarios.csv.gz")
    v_com, v_narr = _leer("verdad/comentarios.csv.gz"), _leer("verdad/narrativas.csv")
    tema_de_narr = dict(zip(v_narr["narrativa_id"], v_narr["tema"]))
    m = com[["comentario_id", "publicacion_id"]].merge(v_com, on="comentario_id")
    m = m[m["narrativa_id"] != "ninguna"]
    m["tema"] = m["narrativa_id"].map(tema_de_narr)
    return m.groupby("publicacion_id")["tema"].agg(lambda s: s.value_counts().index[0]), com


def _correr(nombre, carpeta_datos: Path, modelo_ruido=None, semilla=None):
    """Corre el pipeline + el puntaje oficial. Devuelve una fila de resultados."""
    entrega = TRABAJO / nombre / "entrega"
    comando = [sys.executable, str(RAIZ / "correr.py"), "--datos", str(carpeta_datos),
               "--entrega", str(entrega), "--sin-ia"]
    if modelo_ruido:
        comando += ["--modelo-ruido", str(modelo_ruido)]
    entorno = dict(os.environ)
    if semilla is not None:
        entorno["QUALY_SEMILLA"] = str(semilla)
    print(f"▶ {nombre}", flush=True)
    subprocess.run(comando, cwd=RAIZ, check=True, env=entorno, capture_output=True)
    salida_json = TRABAJO / nombre / "puntaje.json"
    subprocess.run([sys.executable, str(RAIZ / "evaluacion" / "puntuar.py"), "--verdad",
                    str(carpeta_datos / "verdad"), "--entrega", str(entrega), "--json", str(salida_json)],
                   cwd=RAIZ, check=True, capture_output=True)
    r = json.loads(salida_json.read_text(encoding="utf-8"))
    n, c = r["narrativas"], r["coordinacion"]
    verdad_cue = _leer("verdad/cuentas.csv.gz", carpeta_datos)
    fila = {"experimento": nombre, "narrativas_30": n["puntos"], "bcubed_f1": round(n["bcubed_f1"], 3),
            "ruido_f1": round(n["ruido_f1"], 3), "postura_f1": round(n["postura_macro_f1"], 3),
            "origen": round(n["origen"], 3), "coordinacion_25": c["puntos"],
            "cuentas_f1": round(c["f1"], 3), "cuentas_precision": round(c["precision"], 3),
            "cuentas_marcadas": c["cuentas_marcadas"],
            "cuentas_campana_reales": int((verdad_cue["campana_id"] != "").sum()),
            "total_55": round(n["puntos"] + c["puntos"], 2)}
    print("   ", fila, flush=True)
    return fila


def main():
    TRABAJO.mkdir(parents=True, exist_ok=True)
    filas = []
    tema_pub, com = _tema_por_publicacion()
    tema_com = com["publicacion_id"].map(tema_pub).fillna("?")

    # 0) Referencia: el set de práctica completo, con esta misma versión del código
    filas.append(_correr("referencia_practica_completa", PRACTICA))

    # 1) Tema afuera
    for tema in sorted(set(tema_pub)):
        resto, solo = TRABAJO / f"sin_{tema}" / "datos", TRABAJO / f"tema_{tema}" / "datos"
        _subconjunto(resto, (tema_com != tema).to_numpy())
        _subconjunto(solo, (tema_com == tema).to_numpy())
        modelo = TRABAJO / f"sin_{tema}" / "ruido.json"
        subprocess.run([sys.executable, str(RAIZ / "entrenar_ruido.py"), "--datos", str(resto),
                        "--salida", str(modelo)], cwd=RAIZ, check=True, capture_output=True)
        filas.append(_correr(f"tema_afuera_{tema}", solo, modelo_ruido=modelo))

    # 2) Corte temporal: primeros 10 días
    ts = pd.to_datetime(com["fecha"], utc=True, errors="coerce")
    corte = TRABAJO / "primeros_10_dias" / "datos"
    _subconjunto(corte, (ts < ts.min() + pd.Timedelta(days=10)).to_numpy())
    filas.append(_correr("primeros_10_dias", corte))

    # 3) Semillas
    for semilla in (1, 2, 3):
        filas.append(_correr(f"semilla_{semilla}", PRACTICA, semilla=semilla))

    # 4) Datos sucios
    rng = np.random.default_rng(SEMILLA_DATOS)
    sucio = com.copy()
    n = len(sucio)
    sucio.loc[rng.choice(n, n // 100, replace=False), "fecha"] = "fecha rota"
    sucio.loc[rng.choice(n, n // 100, replace=False), "texto"] = ""
    sucio = pd.concat([sucio, sucio.sample(n // 20, random_state=SEMILLA_DATOS)])   # 5% duplicados
    sucio["columna_extra"] = "x"
    sucio = sucio[list(reversed(sucio.columns))]                                     # columnas al revés
    cue = _leer("cuentas.csv.gz")
    cue = cue.drop(index=cue.sample(frac=0.1, random_state=SEMILLA_DATOS).index)    # 10% sin metadata
    carpeta = TRABAJO / "datos_sucios" / "datos"
    _guardar_set(carpeta, sucio, cue, _leer("publicaciones.csv.gz"), _leer("verdad/comentarios.csv.gz"),
                 _leer("verdad/cuentas.csv.gz"), _leer("verdad/narrativas.csv"))
    filas.append(_correr("datos_sucios", carpeta))

    # Guardar resultados
    salida = RAIZ / "experimentos" / "generalizacion.csv"
    with salida.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    print(f"→ {salida.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
