"""Escribe los archivos de entrega/ con el formato EXACTO que lee el jurado.

Pase lo que pase en las otras etapas, esta función tiene que poder escribir
una entrega válida: si falta algo, lo completa con el valor "vacío" permitido
(narrativa vacía = ruido, grupo vacío = cuenta genuina).
"""
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

# Zona horaria argentina (-03:00), la misma que usan los datos
ARGENTINA = timezone(timedelta(hours=-3))
POSTURAS_VALIDAS = {"apoya", "rechaza"}


def ts_a_iso(ts) -> str:
    """Convierte segundos-desde-1970 a texto ISO con zona -03:00 (ej: 2026-09-07T08:15:00-03:00)."""
    if ts is None or pd.isna(ts):
        return ""
    return datetime.fromtimestamp(int(ts), tz=ARGENTINA).isoformat()


def escribir_entrega(carpeta: Path, comentario_ids, cuenta_ids,
                     narrativa=None, postura=None, grupo=None, narrativas=None):
    """Escribe comentarios.csv, narrativas.csv y cuentas.csv.

    comentario_ids: todos los ids de comentarios (cada uno lleva una fila, sí o sí)
    cuenta_ids:     todos los ids de cuentas
    narrativa:      dict {comentario_id: etiqueta}  (si falta un id -> ruido)
    postura:        dict {comentario_id: "apoya"/"rechaza"}
    grupo:          dict {cuenta_id: etiqueta de operación}  (si falta -> genuina)
    narrativas:     lista de dicts con narrativa, titulo, frases_clave, fecha_origen
    """
    carpeta.mkdir(parents=True, exist_ok=True)
    narrativa, postura, grupo = narrativa or {}, postura or {}, grupo or {}

    # --- comentarios.csv --------------------------------------------------------
    com = pd.DataFrame({"comentario_id": list(comentario_ids)})
    com["narrativa"] = com["comentario_id"].map(narrativa).fillna("").astype(str)
    com["postura"] = com["comentario_id"].map(postura).fillna("").astype(str).str.lower()
    # Reglas de formato: la postura solo puede ser apoya/rechaza, y el ruido no lleva postura
    com.loc[~com["postura"].isin(POSTURAS_VALIDAS), "postura"] = ""
    com.loc[com["narrativa"] == "", "postura"] = ""
    com.to_csv(carpeta / "comentarios.csv", index=False)

    # --- narrativas.csv ---------------------------------------------------------
    columnas = ["narrativa", "titulo", "frases_clave", "fecha_origen"]
    usadas = set(com["narrativa"]) - {""}
    filas = [n for n in (narrativas or []) if n.get("narrativa") in usadas]
    tabla = pd.DataFrame(filas, columns=columnas).fillna("")
    # Toda etiqueta usada en comentarios.csv tiene que tener su fila en narrativas.csv
    for faltante in sorted(usadas - set(tabla["narrativa"])):
        tabla.loc[len(tabla)] = [faltante, faltante, "", ""]
    tabla.to_csv(carpeta / "narrativas.csv", index=False)

    # --- cuentas.csv ------------------------------------------------------------
    cue = pd.DataFrame({"cuenta_id": list(cuenta_ids)})
    cue["grupo"] = cue["cuenta_id"].map(grupo).fillna("").astype(str)
    cue.to_csv(carpeta / "cuentas.csv", index=False)

    return {"comentarios": len(com), "narrativas": len(tabla),
            "cuentas_marcadas": int((cue["grupo"] != "").sum())}
