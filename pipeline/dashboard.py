"""Etapa 6: arma un dashboard HTML (un solo archivo, sin internet) con los resultados.

La idea es que el jurado entienda en un minuto:
  - QUÉ circula:      las narrativas, de la más grande a la más chica, con su título.
  - DESDE CUÁNDO:     una mini línea de tiempo por narrativa, con la fecha de origen marcada.
  - QUIÉN LA EMPUJA:  qué operación coordinada publica en cada narrativa, y la red de esas cuentas.

Por qué así:
- Una mini línea por narrativa ("small multiples") en vez de 12 colores en un mismo gráfico:
  12 colores no se distinguen; 12 filas con la misma escala sí.
- La postura es "polaridad" (a favor / en contra): azul y rojo, opuestos.
- La red se dibuja con posiciones calculadas acá (con semilla fija) y el HTML solo las pinta.

Todo el texto que viene de los datos se inserta con textContent (nunca como HTML),
así un comentario con "<script>" no puede romper ni atacar la página.
"""
import json
import logging
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import salida

DIA_S = 86400
VENTANA_RED_S = 30 * 60      # dos cuentas coordinadas "co-actúan" si comentan la misma publicación en 30 min
MIN_COACCIONES = 2           # una línea en la red = al menos 2 co-acciones o respuestas


def _dia(ts):
    return datetime.fromtimestamp(int(ts), tz=salida.ARGENTINA).strftime("%Y-%m-%d")


def _red(com, grupo):
    """Nodos y aristas de la red de cuentas coordinadas, con posiciones fijas (semilla 42)."""
    import igraph as ig
    marcadas = sorted(grupo)
    if not marcadas:
        return [], []
    sub = com[com["cuenta_id"].isin(set(marcadas)) & com["ts"].notna()]
    peso = Counter()
    # Co-acción: misma publicación, menos de 30 minutos de diferencia
    for _, g in sub.groupby("publicacion_id"):
        filas = sorted(zip(g["ts"].astype(int), g["cuenta_id"]))
        for i in range(len(filas)):
            for j in range(i + 1, len(filas)):
                if filas[j][0] - filas[i][0] > VENTANA_RED_S:
                    break
                a, b = filas[i][1], filas[j][1]
                if a != b:
                    peso[(min(a, b), max(a, b))] += 1
    # Respuestas entre cuentas marcadas
    autor = dict(zip(com["comentario_id"], com["cuenta_id"]))
    for c, p in zip(sub["cuenta_id"], sub["respuesta_a"]):
        o = autor.get(p) if isinstance(p, str) else None
        if o and o != c and o in grupo:
            peso[(min(c, o), max(c, o))] += 1
    aristas = [(a, b, w) for (a, b), w in peso.items() if w >= MIN_COACCIONES]

    numero = {c: i for i, c in enumerate(marcadas)}
    G = ig.Graph(n=len(marcadas), edges=[(numero[a], numero[b]) for a, b, _ in aristas])
    # Aristas invisibles dentro de cada operación: mantienen junta a la operación en el dibujo
    por_op = defaultdict(list)
    for c in marcadas:
        por_op[grupo[c]].append(numero[c])
    for miembros in por_op.values():
        for i in range(1, len(miembros)):
            G.add_edge(miembros[0], miembros[i])
    random.seed(42)                       # igraph usa el azar de Python para arrancar el dibujo
    pos = np.array(G.layout_fruchterman_reingold(niter=800).coords)
    pos = (pos - pos.min(0)) / np.maximum(np.ptp(pos, 0), 1e-9)
    nodos = [{"id": c, "op": grupo[c], "x": round(float(x), 4), "y": round(float(y), 4)}
             for c, (x, y) in zip(marcadas, pos)]
    return nodos, [[numero[a], numero[b], int(w)] for a, b, w in aristas]


def construir_datos(com, cuentas, narrativa, postura, grupo, narrativas, operaciones, costo_info, nombre_set):
    """Junta todo lo que el HTML necesita en un diccionario chico (se embebe como JSON)."""
    com = com.copy()
    com["narr"] = com["comentario_id"].map(lambda c: narrativa.get(c, ""))
    com["post"] = com["comentario_id"].map(lambda c: postura.get(c, ""))
    com["op"] = com["cuenta_id"].map(lambda c: grupo.get(c, ""))
    con_fecha = com[com["ts"].notna()]
    t0, t1 = con_fecha["ts"].min(), con_fecha["ts"].max()
    dias = pd.date_range(_dia(t0), _dia(t1), freq="D").strftime("%Y-%m-%d").tolist()
    con_fecha = con_fecha.assign(dia=con_fecha["ts"].map(_dia))

    filas = []
    for n in narrativas:
        etiqueta = n["narrativa"]
        miembros = con_fecha[con_fecha["narr"] == etiqueta]
        if miembros.empty:
            continue
        por_dia = miembros.groupby("dia").size().reindex(dias, fill_value=0).astype(int).tolist()
        ops = Counter(miembros.loc[miembros["op"] != "", "op"])
        origen = n.get("fecha_origen", "")
        try:
            origen_ts = datetime.fromisoformat(origen).timestamp()
        except ValueError:
            origen_ts = None
        filas.append({
            "narrativa": etiqueta, "titulo": n.get("titulo", etiqueta),
            "frases": [f for f in str(n.get("frases_clave", "")).split("|") if f][:6],
            "origen": origen, "origen_dia": _dia(origen_ts) if origen_ts else "",
            # emergente = nace más de 24 h después del primer comentario del set
            "emergente": bool(origen_ts and origen_ts > t0 + DIA_S),
            "total": int(len(miembros)), "apoya": int((miembros["post"] == "apoya").sum()),
            "rechaza": int((miembros["post"] == "rechaza").sum()), "por_dia": por_dia,
            "ops": [{"op": o, "comentarios": int(k)} for o, k in ops.most_common(3)],
        })

    # Operaciones: cuántas cuentas, qué narrativa empujan, con qué postura, antigüedad de las cuentas
    creacion = dict(zip(cuentas["cuenta_id"], pd.to_datetime(cuentas["fecha_creacion"], errors="coerce")))
    inicio = pd.Timestamp(datetime.fromtimestamp(int(t0)).date())
    ops = []
    for o in operaciones:
        cuentas_op = [c for c, g in grupo.items() if g == o["grupo"]]
        suyos = con_fecha[con_fecha["op"] == o["grupo"]]
        narr = Counter(suyos.loc[suyos["narr"] != "", "narr"]).most_common(1)
        principal = narr[0][0] if narr else ""
        en_principal = suyos[suyos["narr"] == principal]
        edades = [(inicio - creacion[c]).days for c in cuentas_op if pd.notna(creacion.get(c, pd.NaT))]
        horas = Counter(pd.to_datetime(suyos["ts"], unit="s", utc=True).dt.tz_convert("America/Argentina/Buenos_Aires").dt.hour)
        ejemplo = en_principal["texto"].value_counts().index[0] if len(en_principal) else ""
        ops.append({
            "op": o["grupo"], "cuentas": len(cuentas_op), "capas": o["capas"],
            "comentarios": int(len(suyos)), "narrativa": principal,
            "porcentaje_en_narrativa": round(100 * len(en_principal) / max(len(suyos), 1)),
            "postura": (en_principal["post"].value_counts().index[0] if len(en_principal) else ""),
            "edad_mediana_dias": int(np.median(edades)) if edades else None,
            "horas_pico": [int(h) for h, _ in horas.most_common(3)], "ejemplo": str(ejemplo)[:220],
        })
    nodos, aristas = _red(com, grupo)
    return {
        "set": nombre_set, "dias": dias, "comentarios": int(len(com)),
        "ruido_pct": round(100 * (com["narr"] == "").mean(), 1),
        "narrativas": filas, "operaciones": ops, "nodos": nodos, "aristas": aristas,
        "cuentas_coordinadas": len(grupo), "cuentas_total": int(com["cuenta_id"].nunique()),
        "costo": {k: costo_info.get(k) for k in ("segundos", "usd_ia", "usd_por_mil_comentarios", "hardware")},
    }


def escribir(ruta: Path, datos: dict):
    """Escribe el HTML con los datos embebidos."""
    plantilla = (Path(__file__).parent / "dashboard_plantilla.html").read_text(encoding="utf-8")
    # "</" se escapa para que ningún texto de los datos pueda cerrar el <script> antes de tiempo
    js = json.dumps(datos, ensure_ascii=False).replace("</", "<\\/")
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(plantilla.replace("__DATOS__", js), encoding="utf-8")
    logging.info("Dashboard: %s", ruta)
