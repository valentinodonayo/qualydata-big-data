"""Diagnóstico: ¿dónde se pierden los puntos? Compara una entrega de práctica con la verdad.

Uso:
    uv run python evaluacion/diagnostico.py --entrega salida/practica

Escribe reportes/diagnostico.md con:
  - puntos perdidos por sub-métrica (con los mismos pesos que evaluacion/puntuar.py),
  - narrativas reales partidas o mezcladas, ruido mal asignado,
  - postura por narrativa (y por qué falla), error de la fecha de origen en horas,
  - coordinación: falsos negativos y positivos por campaña, operaciones partidas o fusionadas.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
DATOS = RAIZ / "datos" / "practica"


def leer(ruta):
    return pd.read_csv(ruta, dtype=str, keep_default_na=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entrega", type=Path, default=RAIZ / "salida" / "practica")
    ap.add_argument("--salida", type=Path, default=RAIZ / "reportes" / "diagnostico.md")
    args = ap.parse_args()

    # Puntaje oficial (el mismo script del jurado)
    tmp = args.entrega / "_puntaje.json"
    subprocess.run([sys.executable, str(RAIZ / "evaluacion/puntuar.py"), "--verdad", str(DATOS / "verdad"),
                    "--entrega", str(args.entrega), "--json", str(tmp)], check=True, capture_output=True)
    r = json.loads(tmp.read_text())
    tmp.unlink()
    n, c = r["narrativas"], r["coordinacion"]

    lineas = ["# Diagnóstico de puntos perdidos (set de práctica)", "",
              f"Entrega analizada: `{args.entrega.relative_to(RAIZ)}`. Total: "
              f"**{n['puntos'] + c['puntos']:.2f} / 55** (narrativas {n['puntos']}, coordinación {c['puntos']}).", "",
              "## 1. Puntos perdidos por sub-métrica", "",
              "| Sub-métrica | Valor | Peso | Puntos perdidos |", "|---|---|---|---|"]
    perdidas = [
        ("Agrupamiento B-cubed F1", n["bcubed_f1"], 0.5 * 30),
        ("Ruido F1", n["ruido_f1"], 0.15 * 30),
        ("Postura macro-F1", n["postura_macro_f1"], 0.2 * 30),
        ("Origen de emergentes", n["origen"], 0.15 * 30),
        ("Cuentas coordinadas F1", c["f1"], 0.7 * 25),
        ("Operaciones B-cubed F1", c["grupos_bcubed_f1"], 0.3 * 25),
    ]
    for nombre, valor, peso in sorted(perdidas, key=lambda x: -(1 - x[1]) * x[2]):
        lineas.append(f"| {nombre} | {valor:.3f} | {peso:.2f} | **{(1 - valor) * peso:.2f}** |")
    lineas.append(f"| **Total** | | | **{sum((1 - v) * p for _, v, p in perdidas):.2f}** |")

    # --- Narrativas ---------------------------------------------------------------
    com = leer(DATOS / "comentarios.csv.gz")
    pred = leer(args.entrega / "comentarios.csv")
    verdad = leer(DATOS / "verdad/comentarios.csv.gz")
    m = com.merge(pred, on="comentario_id").merge(verdad, on="comentario_id", suffixes=("", "_real"))
    m["ts"] = pd.to_datetime(m["fecha"], utc=True, errors="coerce")
    m["es_resp"] = m["respuesta_a"] != ""
    m["n_pred"] = m["narrativa"].replace("", "RUIDO")

    lineas += ["", "## 2. Narrativas: cómo se reparte cada narrativa real", "",
               "| Real | Comentarios | Narrativa predicha principal | % en la principal | % a ruido | % a otras |",
               "|---|---|---|---|---|---|"]
    for real, g in m[m["narrativa_id"] != "ninguna"].groupby("narrativa_id"):
        vc = g["n_pred"].value_counts()
        principal = vc.drop("RUIDO", errors="ignore").index[0]
        p_ruido = vc.get("RUIDO", 0) / len(g)
        p_princ = vc[principal] / len(g)
        lineas.append(f"| {real} | {len(g)} | {principal} | {p_princ:.1%} | {p_ruido:.1%} | {1 - p_princ - p_ruido:.1%} |")

    ruido_real = m["narrativa_id"] == "ninguna"
    ruido_pred = m["narrativa"] == ""
    fn = m[ruido_real & ~ruido_pred]
    fp = m[~ruido_real & ruido_pred]
    lineas += ["", "## 3. Ruido", "",
               f"- Ruido real que quedó dentro de una narrativa (falso negativo): **{len(fn)}** "
               f"({(~fn['es_resp']).mean():.0%} son raíces). Ejemplos: "
               + "; ".join(f"«{t[:60]}»" for t in fn["texto"].head(4)),
               f"- Comentarios de narrativa marcados como ruido (falso positivo): **{len(fp)}** "
               f"({(~fp['es_resp']).mean():.0%} raíces; mediana {fp['texto'].str.split().str.len().median():.0f} palabras). "
               "Ejemplos: " + "; ".join(f"«{t[:60]}»" for t in fp["texto"].head(4))]

    # --- Postura -----------------------------------------------------------------------
    con_narr = m[(m["narrativa_id"] != "ninguna") & (m["narrativa"] != "")]
    lineas += ["", "## 4. Postura por narrativa real", "",
               "| Real | Acierto | Apoya real | Rechaza real | ¿Bando más grande = apoya? |", "|---|---|---|---|---|"]
    for real, g in con_narr.groupby("narrativa_id"):
        acierto = (g["postura"] == g["postura_real"]).mean()
        ap_ = (g["postura_real"] == "apoya").sum()
        re_ = (g["postura_real"] == "rechaza").sum()
        lineas.append(f"| {real} | {acierto:.1%} | {ap_} | {re_} | {'sí' if ap_ >= re_ else '**no**'} |")

    # --- Origen ---------------------------------------------------------------------------
    narr_real = leer(DATOS / "verdad/narrativas.csv")
    narr_pred = leer(args.entrega / "narrativas.csv").set_index("narrativa")
    lineas += ["", "## 5. Fecha de origen (todas las narrativas, no solo emergentes)", "",
               "| Real | Emergente | Origen real | Origen estimado | Error (h, + = tarde) |", "|---|---|---|---|---|"]
    errores = []
    for _, fila in narr_real.iterrows():
        g = m[m["narrativa_id"] == fila["narrativa_id"]]
        principal = g["n_pred"].value_counts().drop("RUIDO", errors="ignore").index[0]
        est = pd.Timestamp(narr_pred.loc[principal, "fecha_origen"])
        real_ts = pd.Timestamp(fila["fecha_origen"])
        err = (est - real_ts).total_seconds() / 3600
        errores.append((fila["emergente"], err))
        lineas.append(f"| {fila['narrativa_id']} | {fila['emergente']} | {fila['fecha_origen'][5:16]} | "
                      f"{narr_pred.loc[principal, 'fecha_origen'][5:16]} | {err:+.2f} |")
    em = [e for es, e in errores if es == "si"]
    lineas.append(f"\nError medio en emergentes: {sum(em) / len(em):+.2f} h (todas dentro de las 3 h que dan puntaje completo).")

    # --- Coordinación ------------------------------------------------------------------------
    vc = leer(DATOS / "verdad/cuentas.csv.gz")
    pc = leer(args.entrega / "cuentas.csv")
    mc = vc.merge(pc, on="cuenta_id")
    act = com.groupby("cuenta_id").agg(comentarios=("comentario_id", "size"),
                                       publicaciones=("publicacion_id", "nunique"))
    lineas += ["", "## 6. Coordinación", "", "Cruce campaña real × operación detectada:", "", "```",
               pd.crosstab(mc["campana_id"].replace("", "genuina"), mc["grupo"].replace("", "-")).to_string(), "```", ""]
    perdidas_c = mc[(mc["campana_id"] != "") & (mc["grupo"] == "")].merge(act, left_on="cuenta_id", right_index=True)
    lineas.append("Falsos negativos por campaña (actividad mediana de las que se escapan):")
    for camp, g in perdidas_c.groupby("campana_id"):
        lineas.append(f"- {camp}: {len(g)} cuentas, mediana {g['comentarios'].median():.0f} comentarios "
                      f"en {g['publicaciones'].median():.0f} publicaciones")
    fps = mc[(mc["campana_id"] == "") & (mc["grupo"] != "")].merge(act, left_on="cuenta_id", right_index=True)
    lineas.append(f"\nFalsos positivos: {len(fps)} cuenta(s): "
                  + ", ".join(f"{r.cuenta_id} ({r.grupo}, {r.comentarios} comentarios)" for r in fps.itertuples()))
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    args.salida.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print("\n".join(lineas))


if __name__ == "__main__":
    main()
