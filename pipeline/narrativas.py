"""Etapa 4: descubrir narrativas, separar el ruido y estimar cuándo nació cada una.

Idea en 4 pasos (todo sin saber de antemano cuántas narrativas hay):

1. MICRO-GRUPOS. Agrupamos los comentarios "con contenido" (raíces y respuestas
   largas) por significado: UMAP resume cada vector de 384 números en 10, y
   HDBSCAN arma grupos chicos y muy puros (cada uno es una forma de decir algo).

2. UNIR CON LOS HILOS. Una misma narrativa queda partida en muchos micro-grupos
   (los que la apoyan, los que la rechazan, distintas formas de decirla).
   Pero una respuesta larga SIEMPRE es de la misma narrativa que su padre
   (100% en práctica). Entonces cada respuesta que cruza de un micro-grupo a
   otro es un "voto" para unirlos. Con esos votos armamos un grafo y Leiden
   encuentra las comunidades = narrativas. No hay umbral a mano.

3. RUIDO. Los comentarios de ruido ("Feliz cumple", "@fulano mirá") nunca
   forman parte de una discusión: no responden ni reciben respuestas largas.
   Un micro-grupo sin ningún voto queda como ruido.

4. HILOS COMPLETOS. Cada hilo (raíz + todas sus respuestas) lleva la narrativa
   que tiene la mayoría de sus comentarios con contenido. Así las respuestas
   cortas ("Tal cual") heredan la narrativa sin necesidad de entenderlas.

Todos los parámetros son RELATIVOS al tamaño de los datos (no números fijos
de práctica), para que funcione igual con el set final.
"""
import hashlib
import logging
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import hilos
from pipeline import texto as tx

SEMILLA = int(os.environ.get("QUALY_SEMILLA", "42"))   # fija; se cambia solo para medir estabilidad
FRACCION_MICRO = 0.001     # un micro-grupo tiene al menos el 0,1% de los textos a agrupar
MINIMO_MICRO = 15          # ...y nunca menos de 15 textos
MIN_SAMPLES = 10           # qué tan "denso" tiene que ser un grupo para HDBSCAN
UMAP_VECINOS, UMAP_DIMENSIONES = 30, 10
VENTANA_ORIGEN_H, MINIMO_ORIGEN = 6, 5   # origen: 1er momento con ≥5 comentarios del núcleo en 6 h


def _reducir_umap(vectores, carpeta_cache: Path):
    """UMAP con cache (tarda ~1-2 min, así que lo guardamos para no repetirlo)."""
    huella = hashlib.sha1(vectores.tobytes() + str(vectores.shape).encode()
                          + f"{UMAP_VECINOS}-{UMAP_DIMENSIONES}-{SEMILLA}".encode()).hexdigest()[:16]
    ruta = carpeta_cache / "umap" / f"{huella}.npy"
    if ruta.exists():
        logging.info("Narrativas: uso UMAP de la cache")
        return np.load(ruta)
    import umap
    logging.info("Narrativas: UMAP sobre %s textos (1-2 min)...", len(vectores))
    X = umap.UMAP(n_neighbors=UMAP_VECINOS, n_components=UMAP_DIMENSIONES, min_dist=0.0,
                  metric="cosine", random_state=SEMILLA,
                  compatibility_layout=True).fit_transform(vectores)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    np.save(ruta, X)
    return X


def _micro_grupos(emb, unicos, carpeta_cache):
    """Paso 1: HDBSCAN sobre UMAP. Devuelve una etiqueta por texto único (-1 = sin grupo)."""
    import hdbscan
    X = _reducir_umap(emb[unicos], carpeta_cache)
    tamano = max(MINIMO_MICRO, int(len(unicos) * FRACCION_MICRO))
    etiquetas = hdbscan.HDBSCAN(min_cluster_size=tamano, min_samples=MIN_SAMPLES).fit_predict(X)

    # Rescate: un texto que HDBSCAN dejó afuera entra al micro-grupo más parecido
    # si se le parece tanto como los propios miembros (umbral = percentil 5 del grupo).
    K = etiquetas.max() + 1
    if K > 0:
        V = emb[unicos]
        centros = np.vstack([V[etiquetas == k].mean(0) for k in range(K)])
        centros /= np.linalg.norm(centros, axis=1, keepdims=True)
        umbral = np.array([np.percentile(V[etiquetas == k] @ centros[k], 5) for k in range(K)])
        sueltos = np.where(etiquetas < 0)[0]
        if len(sueltos):
            sims = V[sueltos] @ centros.T
            mejor = sims.argmax(1)
            acepta = sims[np.arange(len(sueltos)), mejor] >= umbral[mejor]
            etiquetas[sueltos[acepta]] = mejor[acepta]
    logging.info("Narrativas: %s micro-grupos (%.1f%% de textos sin grupo)", K, 100 * np.mean(etiquetas < 0))
    return etiquetas


def _unir_con_hilos(micro_com, pos_padre, es_larga_resp, K):
    """Paso 2: grafo de votos entre micro-grupos + Leiden. Devuelve narrativa por micro-grupo."""
    import igraph as ig
    import leidenalg as la
    hijos = np.where(es_larga_resp & (pos_padre >= 0))[0]
    a, b = micro_com[hijos], micro_com[pos_padre[hijos]]
    ok = (a >= 0) & (b >= 0) & (a != b)
    votos = Counter(zip(np.minimum(a[ok], b[ok]), np.maximum(a[ok], b[ok])))
    grado = np.zeros(K)
    for (x, y), w in votos.items():
        grado[x] += w
        grado[y] += w
    G = ig.Graph(n=K, edges=list(votos.keys()), edge_attrs={"weight": list(votos.values())})
    particion = la.find_partition(G, la.RBConfigurationVertexPartition, weights="weight",
                                  resolution_parameter=1.0, seed=SEMILLA)
    comunidad = np.array(particion.membership)
    # Paso 3: un micro-grupo sin ningún voto no participa de discusiones -> ruido
    comunidad[grado == 0] = -1
    logging.info("Narrativas: %s votos entre micro-grupos -> %s narrativas, %s micro-grupos de ruido",
                 sum(votos.values()), len(set(comunidad) - {-1}), int((grado == 0).sum()))
    return comunidad


def _fecha_origen(ts, sims):
    """Primer momento en que el NÚCLEO del grupo junta ≥ MINIMO_ORIGEN comentarios en la ventana.

    El núcleo = la mitad de los comentarios más parecidos al centro de la narrativa.
    Así un comentario mal asignado que aparece antes no adelanta la fecha.
    """
    ok = ~np.isnan(ts)
    ts, sims = ts[ok], sims[ok]
    if len(ts) == 0:
        return None
    nucleo = np.sort(ts[sims >= np.median(sims)])
    if len(nucleo) < MINIMO_ORIGEN:
        return float(nucleo[0])
    ventana = VENTANA_ORIGEN_H * 3600
    # Para cada comentario i: ¿hay ≥ MINIMO_ORIGEN comentarios en [t_i, t_i + ventana]?
    fin = np.searchsorted(nucleo, nucleo + ventana, side="right")
    candidatos = np.where(fin - np.arange(len(nucleo)) >= MINIMO_ORIGEN)[0]
    return float(nucleo[candidatos[0]] if len(candidatos) else nucleo[0])


def _frases_clave(textos, textos_todos, n=6):
    """Palabras típicas del grupo: frecuentes adentro y poco frecuentes en el resto."""
    def conteo(ts):
        c = Counter()
        for t in ts:
            c.update(set(w for w in tx.para_comparar(t).split() if len(w) > 3))
        return c
    adentro, total = conteo(textos), textos_todos
    puntaje = {w: f * np.log(1 + len(textos) / (1 + total.get(w, 0))) for w, f in adentro.items() if f >= 3}
    # Desempate alfabético: sin él, dos palabras con el mismo puntaje cambian de lugar entre
    # corridas (Python ordena los conjuntos de palabras distinto cada vez que arranca)
    return [w for w, _ in sorted(puntaje.items(), key=lambda x: (-x[1], x[0]))[:n]]


PALABRAS_SEGURAS = 9       # las raíces con ≥ 9 palabras son siempre narrativa (100% en práctica)
RASGOS_RUIDO = ["palabras", "sim1", "margen", "pregunta"]


def rasgos_ruido(final, com, emb, emb_indice, pos_padre):
    """Calcula, para las raíces "dudosas", los rasgos que usa el clasificador de ruido.

    Dudosa = raíz (no responde a nadie), sin respuestas, y con < PALABRAS_SEGURAS palabras.
    Rasgos (ninguno depende del tema, para que sirva con temas nuevos):
      palabras: cuántas palabras tiene
      sim1:     parecido con el centro de la narrativa más cercana
      margen:   diferencia entre la narrativa más cercana y la segunda
      pregunta: si tiene un signo de pregunta
    Devuelve (posiciones_dudosas, tabla_de_rasgos, narrativa_mas_cercana).
    """
    textos = com["texto"].tolist()
    palabras = np.array([len(tx.para_comparar(t).split()) for t in textos])
    tiene_respuestas = np.zeros(len(com), dtype=bool)
    tiene_respuestas[pos_padre[pos_padre >= 0]] = True
    dudosa = (pos_padre < 0) & ~tiene_respuestas & (palabras < PALABRAS_SEGURAS)

    # Centro de cada narrativa, calculado SOLO con comentarios largos (seguros)
    etiquetas = sorted(set(final[(final >= 0) & (palabras >= PALABRAS_SEGURAS)]))
    if len(etiquetas) < 2:
        return np.array([], dtype=int), pd.DataFrame(columns=RASGOS_RUIDO), np.array([])
    centros = np.vstack([emb[emb_indice[(final == k) & (palabras >= PALABRAS_SEGURAS)]].mean(0)
                         for k in etiquetas])
    centros /= np.linalg.norm(centros, axis=1, keepdims=True)

    pos = np.where(dudosa)[0]
    S = emb[emb_indice[pos]] @ centros.T
    orden = np.sort(S, axis=1)
    rasgos = pd.DataFrame({
        "palabras": palabras[pos].astype(float),
        "sim1": orden[:, -1],
        "margen": orden[:, -1] - orden[:, -2],
        "pregunta": np.array(["?" in textos[i] for i in pos], dtype=float),
    })
    return pos, rasgos, np.array(etiquetas)[S.argmax(1)]


def _filtrar_ruido(final, com, emb, emb_indice, pos_padre, modelo_ruido):
    """Decide cuáles raíces dudosas son ruido, con el clasificador entrenado en práctica."""
    pos, rasgos, cercana = rasgos_ruido(final, com, emb, emb_indice, pos_padre)
    if len(pos) == 0:
        return final
    if modelo_ruido:
        # Regresión logística: puntaje = intercepto + suma(coef * rasgo); ruido si puntaje > 0
        coef = np.array([modelo_ruido["coeficientes"][r] for r in RASGOS_RUIDO])
        es_ruido = rasgos[RASGOS_RUIDO].to_numpy() @ coef + modelo_ruido["intercepto"] > 0
    else:
        # Respaldo si no hay modelo entrenado: regla simple de parecido
        es_ruido = rasgos["sim1"].to_numpy() < 0.90
    final = final.copy()
    final[pos[es_ruido]] = -1
    # Las dudosas que NO son ruido pero habían quedado sin narrativa van a la más cercana
    sin_narr = ~es_ruido & (final[pos] < 0)
    final[pos[sin_narr]] = cercana[sin_narr]
    logging.info("Narrativas: %s raíces dudosas, %s marcadas como ruido (%s)", len(pos), int(es_ruido.sum()),
                 "modelo entrenado" if modelo_ruido else "regla de respaldo")
    return final


def detectar(com: pd.DataFrame, emb, emb_indice, carpeta_cache: Path, modelo_ruido=None,
             solo_sin_filtrar=False):
    """Devuelve (narrativa_por_comentario, tabla_narrativas, raiz_por_comentario, extra).

    com: comentarios limpios (comentario_id, respuesta_a, texto, ts, ...)
    emb, emb_indice: salida de embeddings.calcular()
    modelo_ruido: coeficientes del clasificador de ruido (modelos/ruido.json) o None
    solo_sin_filtrar: lo usa el script de entrenamiento (devuelve antes del paso 5)
    """
    ids = com["comentario_id"].tolist()
    textos = com["texto"].tolist()
    pos = {c: i for i, c in enumerate(ids)}
    pos_padre = np.array([pos.get(p, -1) if p is not None and p == p else -1 for p in com["respuesta_a"]])
    es_resp = pos_padre >= 0
    corta = np.array([hilos.es_corta(t) for t in textos])

    # Paso 1: micro-grupos sobre los textos con contenido (raíces + respuestas largas)
    con_contenido = ~(es_resp & corta)
    unicos = np.unique(emb_indice[con_contenido])
    etiqueta_unico = _micro_grupos(emb, unicos, carpeta_cache)
    micro_de_texto = np.full(emb.shape[0], -1)
    micro_de_texto[unicos] = etiqueta_unico
    micro_com = np.where(con_contenido, micro_de_texto[emb_indice], -1)
    K = int(etiqueta_unico.max() + 1)

    # Pasos 2 y 3: unir micro-grupos con los votos de los hilos; los sin votos son ruido
    comunidad = _unir_con_hilos(micro_com, pos_padre, es_resp & ~corta, K) if K else np.array([])
    narr_com = np.where(micro_com >= 0, comunidad[np.maximum(micro_com, 0)] if K else -1, -1)

    # Paso 4: cada hilo lleva la narrativa mayoritaria de sus comentarios con contenido
    raiz = hilos.raices(ids, com["respuesta_a"].tolist())
    hilo = pd.Series([raiz[c] for c in ids])
    votos_hilo = pd.DataFrame({"hilo": hilo, "n": narr_com})
    votos_hilo = votos_hilo[votos_hilo.n >= 0]
    mayoria = votos_hilo.groupby("hilo")["n"].agg(lambda s: s.value_counts().index[0])
    final = hilo.map(mayoria).fillna(-1).astype(int).to_numpy()
    if solo_sin_filtrar:
        return final, pos_padre

    # Paso 5: limpiar el ruido que se coló. Son raíces cortas, sin respuestas,
    # que quedaron dentro de algún micro-grupo. Un clasificador chico decide.
    final = _filtrar_ruido(final, com, emb, emb_indice, pos_padre, modelo_ruido)

    # Etiquetas legibles n01, n02... ordenadas por tamaño (la más grande primero)
    tamanos = Counter(final[final >= 0])
    nombre = {k: f"n{j + 1:02d}" for j, (k, _) in enumerate(tamanos.most_common())}
    narrativa = {c: nombre[k] for c, k in zip(ids, final) if k >= 0}

    # Tabla de narrativas: título provisorio, frases clave y fecha de origen
    todos = Counter()
    for t in textos:
        todos.update(set(w for w in tx.para_comparar(t).split() if len(w) > 3))
    ts = com["ts"].to_numpy(dtype=float)
    tabla = []
    for k, etiqueta in nombre.items():
        miembros = np.where((final == k) & con_contenido)[0]
        V = emb[emb_indice[miembros]]
        centro = V.mean(0)
        centro /= np.linalg.norm(centro)
        sims = V @ centro
        frases = _frases_clave([textos[i] for i in miembros], todos)
        tabla.append({"narrativa": etiqueta, "titulo": " / ".join(frases[:3]),
                      "frases_clave": "|".join(frases), "origen_ts": _fecha_origen(ts[miembros], sims),
                      "comentarios": int(tamanos[k]),
                      # ejemplos representativos (los más cercanos al centro), útiles para la IA
                      "ejemplos": [textos[i] for i in miembros[np.argsort(-sims)[:30]]]})
    logging.info("Narrativas: %s narrativas, %.1f%% de comentarios como ruido",
                 len(tabla), 100 * (1 - len(narrativa) / max(len(ids), 1)))
    # Datos internos que necesita la etapa de postura (micro-grupos y padres)
    extra = {"micro_com": micro_com, "pos_padre": pos_padre, "corta": corta,
             "narr_num": np.array([nombre.get(k, "") for k in final], dtype=object)}
    return narrativa, tabla, raiz, extra
