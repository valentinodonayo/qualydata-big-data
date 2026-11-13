"""Pruebas de estrés: ¿el pipeline aguanta un set DISTINTO al de práctica?

Uso (una prueba por vez; cada una arma su set, corre el pipeline sin IA y lo puntúa):
    python evaluacion/estres.py muestra_25
    python evaluacion/estres.py --lista          # muestra los nombres de las pruebas

Cada prueba escribe:
    salida/estres/<nombre>/datos/      el set armado (formato oficial, con verdad/ si se puede puntuar)
    salida/estres/<nombre>/entrega/    lo que escribió el pipeline
    cache_estres/<nombre>/             caché propia (no toca la caché principal)
y agrega una fila a salida/estres/resultados.jsonl.

Truco para ir rápido: los embeddings de práctica y del final ya están calculados en cache/.
Como el pipeline guarda UN archivo por conjunto de textos, acá armamos ese archivo para
cada set nuevo copiando los vectores que ya existen (si falta alguno, el pipeline lo calcula).
"""
import gzip
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from pipeline import embeddings as emb_mod   # noqa: E402
from pipeline import texto as tx             # noqa: E402

PRACTICA, FINAL = RAIZ / "datos" / "practica", RAIZ / "datos" / "final"
TRABAJO = RAIZ / "salida" / "estres"
CACHES = RAIZ / "cache_estres"
CACHE_PRINCIPAL = RAIZ / "cache"
SEMILLA = 7
PREFIJO_FINAL = "F"       # para que los ids del set final no choquen con los de práctica


def leer(nombre, carpeta=PRACTICA):
    return pd.read_csv(carpeta / nombre, dtype=str, keep_default_na=False)


def escribir(df, ruta: Path):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    if ruta.suffix == ".gz":
        with gzip.open(ruta, "wt", encoding="utf-8") as f:
            df.to_csv(f, index=False)
    else:
        df.to_csv(ruta, index=False)


def practica_completa():
    """Devuelve un diccionario con todas las tablas de práctica (datos + verdad)."""
    return {"com": leer("comentarios.csv.gz"), "cue": leer("cuentas.csv.gz"), "pub": leer("publicaciones.csv.gz"),
            "v_com": leer("verdad/comentarios.csv.gz"), "v_cue": leer("verdad/cuentas.csv.gz"),
            "v_narr": leer("verdad/narrativas.csv")}


def recortar(t, mascara_com):
    """Se queda con los comentarios de la máscara y recorta todo lo demás (cuentas, verdad...)."""
    com = t["com"][mascara_com]
    ids, cuentas = set(com["comentario_id"]), set(com["cuenta_id"])
    v_com = t["v_com"][t["v_com"]["comentario_id"].isin(ids)]
    return {"com": com, "cue": t["cue"][t["cue"]["cuenta_id"].isin(cuentas)],
            "pub": t["pub"][t["pub"]["publicacion_id"].isin(set(com["publicacion_id"]))],
            "v_com": v_com, "v_cue": t["v_cue"][t["v_cue"]["cuenta_id"].isin(cuentas)],
            "v_narr": t["v_narr"][t["v_narr"]["narrativa_id"].isin(set(v_com["narrativa_id"]))]}


def guardar(t, carpeta: Path):
    """Escribe un set en el formato oficial. Las tablas que valen None no se escriben."""
    if carpeta.exists():
        shutil.rmtree(carpeta)
    carpeta.mkdir(parents=True)
    for clave, nombre in (("com", "comentarios.csv.gz"), ("cue", "cuentas.csv.gz"), ("pub", "publicaciones.csv.gz"),
                          ("v_com", "verdad/comentarios.csv.gz"), ("v_cue", "verdad/cuentas.csv.gz"),
                          ("v_narr", "verdad/narrativas.csv")):
        if t.get(clave) is not None:
            escribir(t[clave], carpeta / nombre)
    shutil.copy(PRACTICA / "fuentes.csv", carpeta / "fuentes.csv")


def muestra(t, fraccion):
    """Muestra al azar de PUBLICACIONES (hilos completos), siempre con la misma semilla."""
    pubs = np.array(sorted(t["com"]["publicacion_id"].unique()))
    rng = np.random.default_rng(SEMILLA)
    elegidas = set(rng.choice(pubs, max(1, int(round(len(pubs) * fraccion))), replace=False))
    return recortar(t, t["com"]["publicacion_id"].isin(elegidas).to_numpy())


def tema_por_publicacion(t):
    """Tema de cada publicación = el de la mayoría de sus comentarios con narrativa (de la verdad)."""
    tema_de = dict(zip(t["v_narr"]["narrativa_id"], t["v_narr"]["tema"]))
    m = t["com"][["comentario_id", "publicacion_id"]].merge(t["v_com"], on="comentario_id")
    m = m[m["narrativa_id"] != "ninguna"]
    m["tema"] = m["narrativa_id"].map(tema_de)
    return m.groupby("publicacion_id")["tema"].agg(lambda s: s.value_counts().index[0])


def correr_fechas(t, dias=400):
    """Corre TODAS las fechas +dias y las escribe en UTC con 'Z' (otra zona horaria)."""
    t = {k: v.copy() for k, v in t.items()}
    delta = pd.Timedelta(days=dias)

    def mover(serie):
        return (pd.to_datetime(serie, utc=True) + delta).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    t["com"]["fecha"] = mover(t["com"]["fecha"])
    t["pub"]["fecha"] = mover(t["pub"]["fecha"])
    t["cue"]["fecha_creacion"] = (pd.to_datetime(t["cue"]["fecha_creacion"]) + delta).dt.strftime("%Y-%m-%d")
    # En la verdad dejamos +00:00 (el script oficial lee ISO con cualquier versión de Python)
    t["v_narr"]["fecha_origen"] = (pd.to_datetime(t["v_narr"]["fecha_origen"], utc=True)
                                   + delta).dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    return t


def sin_tildes(texto):
    """'está' -> 'esta' (la ñ se conserva)."""
    t = unicodedata.normalize("NFD", texto.replace("ñ", "\x00").replace("Ñ", "\x01"))
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return unicodedata.normalize("NFC", t.replace("\x00", "ñ").replace("\x01", "Ñ"))


def con_textos(t, funcion):
    t = dict(t)
    t["com"] = t["com"].assign(texto=t["com"]["texto"].map(funcion))
    return t


def practica_mas_final():
    """Práctica + final en un solo set; los ids del final llevan prefijo para no chocar."""
    t = practica_completa()
    com, cue, pub = (leer(n, FINAL) for n in ("comentarios.csv.gz", "cuentas.csv.gz", "publicaciones.csv.gz"))
    for df, columnas in ((com, ["comentario_id", "publicacion_id", "cuenta_id"]), (cue, ["cuenta_id"]),
                         (pub, ["publicacion_id"])):
        for c in columnas:
            df[c] = PREFIJO_FINAL + df[c]
    com.loc[com["respuesta_a"] != "", "respuesta_a"] = PREFIJO_FINAL + com["respuesta_a"]
    # Mezclados por fecha, como llegarían en un set real
    junto = pd.concat([t["com"], com], ignore_index=True)
    junto = junto.iloc[np.argsort(pd.to_datetime(junto["fecha"], utc=True).to_numpy(), kind="stable")]
    t["com"] = junto
    t["cue"] = pd.concat([t["cue"], cue], ignore_index=True)
    t["pub"] = pd.concat([t["pub"], pub], ignore_index=True)
    return t          # la verdad queda la de práctica (del final no hay)


# --- Las pruebas: nombre -> función que devuelve el set (diccionario de tablas) -----------------
def _hospital_5_dias():
    t = practica_completa()
    tema = t["com"]["publicacion_id"].map(tema_por_publicacion(t)).fillna("?")
    ts = pd.to_datetime(t["com"]["fecha"], utc=True)
    return recortar(t, ((tema == "hospital") & (ts < ts.min() + pd.Timedelta(days=5))).to_numpy())


def _sin_campanas():
    t = practica_completa()
    de_campana = set(t["v_cue"][t["v_cue"]["campana_id"] != ""]["cuenta_id"])
    return recortar(t, (~t["com"]["cuenta_id"].isin(de_campana)).to_numpy())


def _sin(t, clave):
    t = dict(t)
    t[clave] = None
    return t


def _sin_hilos(t):
    t = dict(t)
    t["com"] = t["com"].assign(respuesta_a="")
    return t


def _base25():
    return muestra(practica_completa(), 0.25)


PRUEBAS = {
    "referencia": practica_completa,
    "muestra_50": lambda: muestra(practica_completa(), 0.50),
    "muestra_25": _base25,
    "muestra_10": lambda: muestra(practica_completa(), 0.10),
    "muestra_05": lambda: muestra(practica_completa(), 0.05),
    "practica_mas_final": practica_mas_final,
    "sin_campanas": _sin_campanas,
    "hospital_5_dias": _hospital_5_dias,
    "fechas_utc_400": lambda: correr_fechas(practica_completa()),
    "mayusculas_25": lambda: con_textos(_base25(), str.upper),
    "sin_tildes_25": lambda: con_textos(_base25(), sin_tildes),
    "emoji_25": lambda: con_textos(_base25(), lambda s: s + " 🔥"),
    "sin_cuentas_25": lambda: _sin(_base25(), "cue"),
    "cuentas_vacio_25": lambda: dict(_base25(), cue=leer("cuentas.csv.gz").iloc[0:0]),
    "sin_hilos_25": lambda: _sin_hilos(_base25()),
    "solo_500": lambda: recortar(practica_completa(), np.arange(200000) < 500),
}


# --- Caché de embeddings armada con vectores ya calculados ---------------------------------------
def _vectores_conocidos():
    """{texto normalizado: vector} con todo lo que ya se calculó para práctica y final."""
    conocidos = {}
    for carpeta in (PRACTICA, FINAL):
        textos = leer("comentarios.csv.gz", carpeta)["texto"]
        unicos = np.unique(np.array([tx.para_embeddings(x) for x in textos], dtype=object)).tolist()
        huella = hashlib.sha1((emb_mod.MODELO + "\n" + "\n".join(unicos)).encode("utf-8")).hexdigest()[:16]
        ruta = CACHE_PRINCIPAL / "embeddings" / f"{huella}.npy"
        if not ruta.exists():
            print(f"   (no hay embeddings guardados de {carpeta.name})")
            continue
        matriz = np.load(ruta, mmap_mode="r")
        for u, v in zip(unicos, matriz):
            conocidos.setdefault(u, v)
    return conocidos


def precalentar_embeddings(com, cache: Path):
    """Si TODOS los textos del set ya tienen vector, escribe el archivo que el pipeline va a buscar."""
    unicos = np.unique(np.array([tx.para_embeddings(x) for x in com["texto"]], dtype=object)).tolist()
    huella = hashlib.sha1((emb_mod.MODELO + "\n" + "\n".join(unicos)).encode("utf-8")).hexdigest()[:16]
    ruta = cache / "embeddings" / f"{huella}.npy"
    if ruta.exists():
        return True
    conocidos = _vectores_conocidos()
    if any(u not in conocidos for u in unicos):
        return False
    ruta.parent.mkdir(parents=True, exist_ok=True)
    np.save(ruta, np.vstack([conocidos[u] for u in unicos]).astype(np.float32))
    return True


# --- Correr y puntuar -----------------------------------------------------------------------------
def correr(nombre, datos: Path, cache: Path, entrega: Path):
    """Corre el pipeline midiendo tiempo y memoria máxima (con /usr/bin/time -l de macOS)."""
    comando = ["/usr/bin/time", "-l", sys.executable, str(RAIZ / "correr.py"), "--datos", str(datos),
               "--entrega", str(entrega), "--cache", str(cache), "--sin-ia"]
    inicio = time.time()
    r = subprocess.run(comando, cwd=RAIZ, capture_output=True, text=True)
    segundos = time.time() - inicio
    m = re.search(r"(\d+)\s+maximum resident set size", r.stderr)
    (TRABAJO / nombre / "stdout.txt").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr, encoding="utf-8")
    return {"codigo_salida": r.returncode, "segundos": round(segundos, 1),
            "memoria_gb": round(int(m.group(1)) / 1e9, 2) if m else None}


def puntuar(verdad: Path, entrega: Path, salida_json: Path):
    r = subprocess.run([sys.executable, str(RAIZ / "evaluacion" / "puntuar.py"), "--verdad", str(verdad),
                        "--entrega", str(entrega), "--json", str(salida_json)], cwd=RAIZ,
                       capture_output=True, text=True)
    if r.returncode != 0:
        return {"error_puntaje": (r.stderr or r.stdout)[-300:]}
    p = json.loads(salida_json.read_text(encoding="utf-8"))
    n, c = p["narrativas"], p["coordinacion"]
    return {"bcubed": round(n["bcubed_f1"], 3), "bcubed_p": round(n["bcubed_precision"], 3),
            "bcubed_r": round(n["bcubed_recall"], 3), "ruido_f1": round(n["ruido_f1"], 3),
            "postura": round(n["postura_macro_f1"], 3), "origen": round(n["origen"], 3),
            "cuentas_f1": round(c["f1"], 3), "cuentas_precision": round(c["precision"], 3),
            "cuentas_recall": round(c["recall"], 3), "grupos": round(c["grupos_bcubed_f1"], 3),
            "total_55": round(n["puntos"] + c["puntos"], 2)}


def una_prueba(nombre):
    carpeta = TRABAJO / nombre
    datos, entrega, cache = carpeta / "datos", carpeta / "entrega", CACHES / nombre
    print(f"▶ {nombre}: armando el set...", flush=True)
    t = PRUEBAS[nombre]()
    guardar(t, datos)
    if nombre == "cuentas_vacio_25":
        pass   # cuentas.csv.gz queda solo con el encabezado
    fila = {"prueba": nombre, "comentarios": len(t["com"]), "publicaciones": t["com"]["publicacion_id"].nunique()}
    fila["emb_precalentados"] = precalentar_embeddings(t["com"], cache)
    # UMAP guardado: solo sirve si los vectores son exactamente los mismos (sets con todos los textos)
    (cache / "umap").mkdir(parents=True, exist_ok=True)
    for f in (CACHE_PRINCIPAL / "umap").glob("*.npy"):
        if not (cache / "umap" / f.name).exists():
            shutil.copy(f, cache / "umap" / f.name)
    if entrega.exists():
        shutil.rmtree(entrega)
    fila.update(correr(nombre, datos, cache, entrega))

    # Qué salió
    log = (cache / "reportes" / "log.txt").read_text(encoding="utf-8") if (cache / "reportes" / "log.txt").exists() else ""
    fila["log_problemas"] = [l[:200] for l in log.splitlines() if re.search(r"✖|Falló|Error|Hubo errores|respaldo|casi no hay", l)]
    fila["umap_cache"] = "uso UMAP de la cache" in log
    if (entrega / "narrativas.csv").exists():
        fila["narrativas_encontradas"] = len(pd.read_csv(entrega / "narrativas.csv"))
        cue = pd.read_csv(entrega / "cuentas.csv", dtype=str, keep_default_na=False)
        fila["cuentas_marcadas"] = int((cue["grupo"] != "").sum())
        fila["operaciones"] = cue[cue["grupo"] != ""]["grupo"].value_counts().to_dict()
        ent = pd.read_csv(entrega / "comentarios.csv", dtype=str, keep_default_na=False)
        fila["pct_ruido_entregado"] = round(100 * float((ent["narrativa"] == "").mean()), 1)
    if t.get("v_narr") is not None:
        fila["narrativas_reales"] = len(t["v_narr"])
        fila["cuentas_campana_reales"] = int((t["v_cue"]["campana_id"] != "").sum())
        fila["pct_ruido_real"] = round(100 * float((t["v_com"]["narrativa_id"] == "ninguna").mean()), 1)
        fila.update(puntuar(datos / "verdad", entrega, carpeta / "puntaje.json"))
    with (TRABAJO / "resultados.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(fila, ensure_ascii=False) + "\n")
    print("   ", json.dumps(fila, ensure_ascii=False), flush=True)
    return fila


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] == "--lista":
        print("\n".join(PRUEBAS))
        sys.exit(0)
    TRABAJO.mkdir(parents=True, exist_ok=True)
    for n in sys.argv[1:]:
        (TRABAJO / n).mkdir(parents=True, exist_ok=True)
        una_prueba(n)
