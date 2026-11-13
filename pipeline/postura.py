"""Etapa 4b: postura de cada comentario (apoya / rechaza su narrativa).

Lo que medimos en práctica:
- Cada micro-grupo es "puro" en postura (99,7%): o todos apoyan o todos rechazan.
- Las respuestas largas SIEMPRE contradicen a su padre (99,9% de las veces une
  micro-grupos de bandos opuestos).

Entonces, dentro de cada narrativa:
1. Armamos un grafo entre micro-grupos: cada respuesta larga dice "estos dos
   están en bandos opuestos". Lo partimos en 2 bandos (como pintar un mapa con
   2 colores tratando de que los vecinos queden de distinto color).
2. Cada raíz toma el bando de su micro-grupo.
3. Cada respuesta toma su postura del padre: corta = igual, larga = la opuesta.
4. Falta saber cuál de los dos bandos es "apoya". Eso lo decide la IA (ia.py)
   mirando ejemplos de cada bando; si no hay IA, se elige el bando más grande.
"""
import logging
import os
from collections import defaultdict

import numpy as np

from pipeline import hilos


def _dos_bandos(nodos, aristas):
    """Divide los micro-grupos de una narrativa en 2 bandos (0 y 1).

    aristas: dict {(a, b): peso} = cuántas respuestas largas unen a y b.
    Busca que el peso de las aristas que CRUZAN de bando sea el máximo
    (problema de "corte máximo"): primero un recorrido que alterna colores por
    las aristas más pesadas, y después mejora local (cambiar de bando a un nodo
    si así cruzan más aristas) hasta que no haya más mejoras.
    """
    vecinos = defaultdict(dict)
    for (a, b), w in aristas.items():
        vecinos[a][b] = vecinos[a].get(b, 0) + w
        vecinos[b][a] = vecinos[b].get(a, 0) + w

    bando = {}
    # 1) Coloreo inicial: arrancamos por el nodo con más peso y alternamos
    for inicio in sorted(nodos, key=lambda n: -sum(vecinos[n].values())):
        if inicio in bando:
            continue
        bando[inicio] = 0
        pila = [inicio]
        while pila:
            v = pila.pop()
            for w, _ in sorted(vecinos[v].items(), key=lambda t: -t[1]):
                if w in nodos and w not in bando:
                    bando[w] = 1 - bando[v]
                    pila.append(w)

    # 2) Mejora local: cambiar de bando si aumenta el peso que cruza
    for _ in range(50):
        cambio = False
        for v in nodos:
            mismo = sum(p for w, p in vecinos[v].items() if w in nodos and bando[w] == bando[v])
            otro = sum(p for w, p in vecinos[v].items() if w in nodos and bando[w] != bando[v])
            if mismo > otro:
                bando[v] = 1 - bando[v]
                cambio = True
        if not cambio:
            break
    return bando


def calcular_bandos(com, emb, emb_indice, extra):
    """Devuelve (bando_por_comentario, resumen_por_narrativa).

    bando_por_comentario: array con 0/1 para las raíces con narrativa (-1 si no aplica).
    resumen_por_narrativa: {narrativa: {"tamano": [n0, n1], "ejemplos": [[...], [...]]}}
    """
    micro = extra["micro_com"]
    padre = extra["pos_padre"]
    narr = extra["narr_num"]
    corta = extra["corta"]
    textos = com["texto"].tolist()
    n = len(com)
    bando_com = np.full(n, -1)
    resumen = {}

    # Aristas entre micro-grupos (respuestas largas), separadas por narrativa
    hijos = np.where((padre >= 0) & ~corta)[0]
    aristas_por_narr = defaultdict(lambda: defaultdict(int))
    for h in hijos:
        a, b = micro[h], micro[padre[h]]
        if a >= 0 and b >= 0 and a != b and narr[h] and narr[h] == narr[padre[h]]:
            aristas_por_narr[narr[h]][(min(a, b), max(a, b))] += 1

    es_raiz = padre < 0
    for etiqueta in sorted(set(narr) - {""}):
        en_narr = narr == etiqueta
        nodos = set(micro[en_narr & (micro >= 0)])
        bando = _dos_bandos(nodos, aristas_por_narr[etiqueta])

        # Raíces de la narrativa: bando de su micro-grupo
        raices = np.where(en_narr & es_raiz)[0]
        con_micro = raices[micro[raices] >= 0]
        bando_com[con_micro] = [bando.get(m, 0) for m in micro[con_micro]]

        # Raíces sin micro-grupo: bando del micro-grupo más parecido de esta narrativa
        sin_micro = raices[micro[raices] < 0]
        if len(sin_micro) and nodos:
            lista = sorted(nodos)
            centros = np.vstack([emb[emb_indice[en_narr & (micro == m)]].mean(0) for m in lista])
            centros /= np.linalg.norm(centros, axis=1, keepdims=True)
            mas_cerca = (emb[emb_indice[sin_micro]] @ centros.T).argmax(1)
            bando_com[sin_micro] = [bando[lista[j]] for j in mas_cerca]

        # Ejemplos de cada bando para la IA: las 12 raíces largas MÁS CENTRALES (más parecidas al
        # centro del bando), sin repetir texto. Son estables: si otra máquina calcula embeddings con
        # decimales apenas distintos, salen los mismos ejemplos y la caché de la IA sigue sirviendo.
        ejemplos = []
        for b in (0, 1):
            pos = np.where(en_narr & es_raiz & (bando_com == b))[0]
            largos = np.array([i for i in pos if len(textos[i].split()) >= 8] or list(pos), dtype=int)
            elegidos = []
            if len(largos):
                V = emb[emb_indice[largos]]
                centro = V.mean(0)
                orden = sorted(range(len(largos)), key=lambda j: (-round(float(V[j] @ centro), 4), textos[largos[j]]))
                vistos = set()
                for j in orden:
                    t = textos[largos[j]]
                    if t not in vistos:
                        vistos.add(t)
                        elegidos.append(t)
                    if len(elegidos) == 12:
                        break
            ejemplos.append(elegidos)
        tamano = [int(((bando_com == b) & en_narr).sum()) for b in (0, 1)]
        resumen[etiqueta] = {"tamano": tamano, "ejemplos": ejemplos}
    logging.info("Postura: bandos armados en %s narrativas", len(resumen))
    return bando_com, resumen


def posturas(com, bando_com, apoya_es, extra):
    """Convierte bandos en apoya/rechaza y completa las respuestas por el hilo.

    apoya_es: {narrativa: 0 o 1} = qué bando es el que apoya la narrativa.
    """
    ids = com["comentario_id"].tolist()
    narr = extra["narr_num"]
    postura_raiz = {}
    for i in np.where(bando_com >= 0)[0]:
        postura_raiz[ids[i]] = "apoya" if bando_com[i] == apoya_es.get(narr[i], 0) else "rechaza"
    todas = hilos.posturas_por_hilo(ids, com["respuesta_a"].tolist(), com["texto"].tolist(), postura_raiz)
    # Solo los comentarios que tienen narrativa llevan postura
    resultado = {}
    for i, c in enumerate(ids):
        if narr[i] != "" and todas.get(c):
            resultado[c] = todas[c]
    return resultado


import re as _re

_LIMPIAR = _re.compile(r"@\w+|#\w+|https?://\S+|www\.\S+")
_PEDAZOS = _re.compile(r"[.:,;!?]")
_REPETIDAS = _re.compile(r"(\w)\1{2,}")      # "precioooo" -> "precio"


def titulos_representativos(com, emb, emb_indice, extra, postura_com, min_palabras=6, max_palabras=20):
    """Título sin IA: el comentario que mejor resume lo que AFIRMA cada narrativa.

    Para cada narrativa toma las raíces que la apoyan, de largo razonable, calcula su
    "centro" de significado y elige la más cercana a ese centro (la más típica).
    Le saca menciones, hashtags y links. Devuelve {narrativa: titulo}.
    """
    narr = extra["narr_num"]
    padre = extra["pos_padre"]
    ids = com["comentario_id"].tolist()
    textos = com["texto"].tolist()
    titulos = {}
    for etiqueta in sorted(set(narr) - {""}):
        limpio = {}
        for i in np.where((narr == etiqueta) & (padre < 0))[0]:
            if postura_com.get(ids[i]) != "apoya":
                continue
            t = " ".join(_LIMPIAR.sub(" ", str(textos[i])).split())
            if min_palabras <= len(t.split()) <= max_palabras:
                limpio[i] = t
        if not limpio:
            continue
        pos = np.array(sorted(limpio))
        V = emb[emb_indice[pos]]
        centro = V.mean(0)
        # Entre los 30 más típicos, el más "limpio": el que tiene menos pedazos sueltos
        # (muletillas tipo "Escuchen:", "ojo", "Triste" agregan comas, puntos y dos puntos)
        cercanos = pos[np.argsort(-(V @ centro))[:30]]
        pedazos = lambda i: len([x for x in _PEDAZOS.split(limpio[i]) if x.strip()])
        mejor = min(cercanos, key=lambda i: (pedazos(i), list(cercanos).index(i)))
        t = _REPETIDAS.sub(r"\1", limpio[mejor]).strip(" .,;")
        if t.isupper():          # TODO EN MAYÚSCULAS se lee como un grito: lo bajamos
            t = t.lower()
        titulos[etiqueta] = (t[0].upper() + t[1:])[:120]
    return titulos
