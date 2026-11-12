"""Revisa que la carpeta entrega/ cumpla el formato oficial del desafío.

Se corre al final de cada corrida (y en los tests). Devuelve una lista de problemas
en castellano; si la lista está vacía, la entrega es válida.

Reglas que revisa (sacadas del README oficial):
- Están los 4 archivos: comentarios.csv, narrativas.csv, cuentas.csv, costo.json.
- Cada CSV tiene exactamente las columnas pedidas.
- Hay una fila por cada comentario y por cada cuenta, sin repetidos.
- La postura es "apoya", "rechaza" o vacía (y vacía solo si es ruido).
- Toda narrativa usada en comentarios.csv tiene su fila en narrativas.csv.
- fecha_origen es una fecha ISO con zona horaria.
- costo.json tiene los campos pedidos.
"""
import csv
import json
from datetime import datetime
from pathlib import Path

COLUMNAS = {
    "comentarios.csv": ["comentario_id", "narrativa", "postura"],
    "narrativas.csv": ["narrativa", "titulo", "frases_clave", "fecha_origen"],
    "cuentas.csv": ["cuenta_id", "grupo"],
}
CAMPOS_COSTO = ["comentarios_procesados", "segundos", "tokens_ia", "usd_ia",
                "usd_por_mil_comentarios", "hardware", "almacenamiento", "notas"]
RUIDO = {"", "ninguna"}


def _leer(ruta: Path):
    with ruta.open(encoding="utf-8", newline="") as f:
        lector = csv.DictReader(f)
        return lector.fieldnames or [], list(lector)


def validar_entrega(carpeta: Path, comentario_ids=None, cuenta_ids=None):
    """Devuelve la lista de problemas encontrados (vacía = todo bien)."""
    carpeta = Path(carpeta)
    problemas = []
    tablas = {}
    for nombre, columnas in COLUMNAS.items():
        ruta = carpeta / nombre
        if not ruta.exists():
            problemas.append(f"falta {nombre}")
            continue
        encabezado, filas = _leer(ruta)
        if encabezado != columnas:
            problemas.append(f"{nombre} tiene columnas {encabezado}, se esperaban {columnas}")
        tablas[nombre] = filas

    com = tablas.get("comentarios.csv", [])
    ids = [f.get("comentario_id", "") for f in com]
    if len(ids) != len(set(ids)):
        problemas.append("comentarios.csv tiene comentario_id repetidos")
    if comentario_ids is not None and set(ids) != set(comentario_ids):
        problemas.append("comentarios.csv no tiene exactamente un renglón por comentario del set")
    for f in com:
        postura, narr = f.get("postura", ""), f.get("narrativa", "")
        if postura not in {"apoya", "rechaza", ""}:
            problemas.append(f"postura inválida '{postura}' en {f.get('comentario_id')}")
            break
        if narr in RUIDO and postura:
            problemas.append(f"el comentario {f.get('comentario_id')} es ruido pero tiene postura")
            break

    cue = tablas.get("cuentas.csv", [])
    ids_cuentas = [f.get("cuenta_id", "") for f in cue]
    if len(ids_cuentas) != len(set(ids_cuentas)):
        problemas.append("cuentas.csv tiene cuenta_id repetidos")
    if cuenta_ids is not None and not set(cuenta_ids) <= set(ids_cuentas):
        problemas.append("cuentas.csv no tiene todas las cuentas del set")

    narr = tablas.get("narrativas.csv", [])
    declaradas = {f.get("narrativa", "") for f in narr}
    usadas = {f.get("narrativa", "") for f in com} - RUIDO
    if usadas - declaradas:
        problemas.append(f"narrativas sin fila en narrativas.csv: {sorted(usadas - declaradas)[:5]}")
    for f in narr:
        try:
            fecha = datetime.fromisoformat(f.get("fecha_origen", ""))
            if fecha.tzinfo is None:
                problemas.append(f"fecha_origen sin zona horaria en {f.get('narrativa')}")
        except ValueError:
            problemas.append(f"fecha_origen inválida en {f.get('narrativa')}: '{f.get('fecha_origen')}'")

    ruta_costo = carpeta / "costo.json"
    if not ruta_costo.exists():
        problemas.append("falta costo.json")
    else:
        try:
            costo = json.loads(ruta_costo.read_text(encoding="utf-8"))
            faltan = [c for c in CAMPOS_COSTO if c not in costo]
            if faltan:
                problemas.append(f"a costo.json le faltan los campos {faltan}")
        except json.JSONDecodeError:
            problemas.append("costo.json no es un JSON válido")
    return problemas
