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
   mirando ejemplos de cada bando; si no hay IA, se elige el bando más grande, salvo que estén
   parejos: ahí desempata una señal local (senal_local: qué bando "refuta").
"""
import logging
import os
import re as _re
from collections import defaultdict

import numpy as np

from pipeline import hilos

_REPETIDAS = _re.compile(r"(\w)\1{2,}")      # "precioooo" -> "precio"


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


def _clave_texto(t):
    """Texto reducido a lo esencial para detectar casi-copias: minúsculas, sin signos ni emojis,
    y sin letras repetidas ("nadaaaa" = "nada")."""
    t = _REPETIDAS.sub(r"\1", str(t).lower())
    return " ".join(_re.sub(r"[^a-záéíóúüñ0-9 ]", " ", t).split())


def _ejemplos_variados(pos, micro, textos, emb, emb_indice, cuantos=12, min_palabras=8):
    """Ejemplos VARIADOS de un bando: el comentario más típico de cada micro-grupo.

    pos: posiciones de las raíces del bando. Cada micro-grupo es una forma distinta de decir algo;
    se recorren del más grande al más chico tomando un ejemplo de cada uno (y una segunda vuelta
    si hay pocos micro-grupos), salteando casi-copias. Todo con desempates fijos (tamaño, texto)
    para que salgan los mismos ejemplos en cualquier máquina.
    """
    por_micro = defaultdict(list)
    for i in pos:
        if micro[i] >= 0 and len(textos[i].split()) >= min_palabras:
            por_micro[int(micro[i])].append(int(i))
    candidatos = []          # por micro-grupo: sus textos ordenados del más típico al menos típico
    for miembros in por_micro.values():
        V = emb[emb_indice[np.array(miembros)]]
        centro = V.mean(0)
        orden = sorted(range(len(miembros)), key=lambda j: (-round(float(V[j] @ centro), 4), textos[miembros[j]]))
        candidatos.append((len(miembros), [textos[miembros[j]] for j in orden]))
    candidatos.sort(key=lambda c: (-c[0], c[1][0]))
    elegidos, vistos = [], set()
    for vuelta in range(2):
        for _, lista in candidatos:
            if len(elegidos) == cuantos:
                break
            # en la 1ª vuelta el más típico; en la 2ª, el siguiente que no sea casi-copia
            for t in lista[:1] if vuelta == 0 else lista[1:40]:
                if _clave_texto(t) not in vistos:
                    vistos.add(_clave_texto(t))
                    elegidos.append(t)
                    break
    return elegidos


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
        # "ejemplos" (los más centrales) suelen ser 12 copias casi iguales de UNA misma frase: sirven
        # para reconocer el bando en la caché, pero le muestran poco a la IA. "variados" son los que
        # lee la IA: una frase por cada forma distinta de decirlo (micro-grupo) dentro del bando.
        variados = [_ejemplos_variados(np.where(en_narr & es_raiz & (bando_com == b))[0], micro, textos,
                                       emb, emb_indice) for b in (0, 1)]
        resumen[etiqueta] = {"tamano": tamano, "ejemplos": ejemplos, "variados": variados}
    logging.info("Postura: bandos armados en %s narrativas", len(resumen))
    return bando_com, resumen


# --- Señal LOCAL de polaridad (sin IA): ¿qué bando "refuta"? -------------------------------------
# Problema: sin respuestas de IA solo queda "el bando más grande apoya", que es una moneda al aire
# cuando los bandos están ~50/50. Idea: el bando que RECHAZA una narrativa habla distinto, sea cual
# sea el tema: desmiente ("es mentira", "no hubo", "ya lo desmintieron"), relativiza ("no es para
# tanto") o reta al otro ("dejen de inventar", "muestren pruebas", "hay que informarse").
# Se mide de dos formas independientes y solo se vota si alguna supera su umbral y no se contradicen:
#   1. LÉXICO: fracción de textos distintos del bando con alguna expresión de refutación.
#   2. PROTOTIPOS: con el mismo modelo de embeddings del pipeline, cuánto se parece el bando a
#      frases genéricas de desmentida más que a frases genéricas de denuncia/reclamo.
# Medido (evaluacion/medir_polaridad_local.py): práctica 12 aciertos, 0 errores, 1 abstención;
# set final (otros temas, comparado con la entrega) 10 aciertos, 0 errores, 2 abstenciones.
# Limitación conocida: en narrativas "positivas" (promesas: "el litio trae trabajo") el que rechaza
# es el que se queja, y la señal se achica o se da vuelta: por eso se abstiene y nunca decide sola.
_REFUTACION = _re.compile(
    r"\b(mentira|fals[oa]|truch[oa]|verso|humo|invent\w*|exager\w*|desmint\w*|difam\w*|no es cierto|no es asi|"
    r"no es (para|pa) tanto|nada (que|q) ver|dejen de|pruebas|informarse|informense|lean bien|sin saber|"
    r"no hubo|no hay ningun\w*|nadie va a|siempre los mismos|critic\w*|culpan|se quejan|quejarse|obvio)\b")
FRASES_REFUTAN = [
    "eso es mentira", "no es así, están exagerando", "muestren pruebas o dejen de difamar", "dejen de inventar",
    "eso ya se desmintió", "no es para tanto", "la cosa no es como la cuentan", "eso lo dicen los que no saben",
    "es falso, no pasó nada de eso", "hay que informarse antes de hablar", "siempre los mismos criticando todo",
    "no tienen razón, ya está solucionado"]
FRASES_AFIRMAN = [
    "es una vergüenza", "nadie hace nada", "me dijeron que va a pasar", "hay que hacer algo ya", "nos están robando",
    "lo están ocultando", "esto es un desastre", "no se puede vivir así", "nos mienten en la cara",
    "es lo mejor que nos pasó", "hace falta que lo hagan", "cada vez estamos peor"]
UMBRAL_LEXICO = 0.10       # diferencia mínima entre bandos en la fracción de textos con refutación
UMBRAL_PROTOTIPOS = 0.01   # diferencia mínima entre bandos en parecido a los prototipos


def fraccion_refutacion(textos):
    """Fracción de textos DISTINTOS (sin contar copias) que tienen alguna expresión de refutación."""
    from pipeline import texto as tx
    distintos = {tx.para_comparar(t) for t in textos}
    if not distintos:
        return 0.0
    return sum(bool(_REFUTACION.search(t)) for t in distintos) / len(distintos)


def direccion_refutacion(carpeta_cache):
    """Vector (largo 1) que apunta "hacia la refutación": centro de las frases que desmienten menos
    centro de las frases que denuncian. Usa el mismo modelo local (y la misma caché) que el pipeline."""
    from pipeline import embeddings
    matriz, indice = embeddings.calcular(FRASES_REFUTAN + FRASES_AFIRMAN, carpeta_cache)
    V = matriz[indice]
    w = V[:len(FRASES_REFUTAN)].mean(0) - V[len(FRASES_REFUTAN):].mean(0)
    return w / np.linalg.norm(w)


def voto_local(dif_lexico, dif_prototipos):
    """Junta las dos medidas. Las dos vienen como "cuánto MÁS refutador es el bando 0 que el 1".

    Devuelve qué bando APOYA (0 o 1), o None si se abstiene: ninguna medida supera su umbral,
    o las dos lo superan pero apuntan a bandos distintos.
    """
    votos = set()
    if abs(dif_lexico) >= UMBRAL_LEXICO:
        votos.add(1 if dif_lexico > 0 else 0)        # el bando 0 refuta más -> apoya el 1
    if abs(dif_prototipos) >= UMBRAL_PROTOTIPOS:
        votos.add(1 if dif_prototipos > 0 else 0)
    return votos.pop() if len(votos) == 1 else None


def senal_local(com, emb, emb_indice, extra, bando_com, direccion):
    """Para cada narrativa: {"voto": bando que apoya (0/1) o None, "lexico": dif, "prototipos": dif}.

    Mira solo las RAÍCES de cada bando (las respuestas heredan la postura por el hilo) y solo textos
    distintos, para que una frase copiada mil veces no pese mil veces.
    direccion: lo que devuelve direccion_refutacion().
    """
    narr = extra["narr_num"]
    textos = com["texto"].tolist()
    resultado = {}
    for etiqueta in sorted(set(narr) - {""}):
        pos = [np.where((narr == etiqueta) & (bando_com == b))[0] for b in (0, 1)]
        if not len(pos[0]) or not len(pos[1]):
            resultado[etiqueta] = {"voto": None, "lexico": 0.0, "prototipos": 0.0}
            continue
        lexico = [fraccion_refutacion(textos[i] for i in p) for p in pos]
        # Centro de significado de cada bando (sobre textos distintos) proyectado en la dirección
        centro = [emb[np.unique(emb_indice[p])].mean(0) for p in pos]
        dif_lexico = float(lexico[0] - lexico[1])
        dif_prototipos = float((centro[0] - centro[1]) @ direccion)
        resultado[etiqueta] = {"voto": voto_local(dif_lexico, dif_prototipos),
                               "lexico": round(dif_lexico, 3), "prototipos": round(dif_prototipos, 4)}
    return resultado


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


_LIMPIAR = _re.compile(r"@\w+|#\w+|https?://\S+|www\.\S+")
_PEDAZOS = _re.compile(r"[.:,;!?]")


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


def bando_de_operaciones(com, bando_com, extra, grupo, min_comentarios=10):
    """Señal extra para decidir qué bando apoya: ¿de qué lado escriben las cuentas coordinadas?

    En práctica, las 3 campañas coordinadas escriben A FAVOR de la narrativa que empujan
    (verdad/campanas.csv). Entonces, si una operación concentra sus comentarios en una narrativa,
    el bando donde caen esos comentarios es un buen candidato a ser el que "apoya".

    grupo: {cuenta_id: "op01", ...} (lo que devuelve coordinacion.detectar).
    Devuelve {narrativa: [comentarios en bando 0, comentarios en bando 1]} contando, para cada
    operación, solo SU narrativa principal (donde más raíces escribió) y solo si escribió ahí
    al menos `min_comentarios` raíces. Las narrativas sin operación no aparecen.
    """
    narr = extra["narr_num"]
    cuentas = com["cuenta_id"].tolist()
    # Por operación: cuántas raíces (con bando) tiene en cada narrativa y bando
    conteo = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for i in np.where(bando_com >= 0)[0]:
        op = grupo.get(cuentas[i])
        if op:
            conteo[op][narr[i]][int(bando_com[i])] += 1
    votos = {}
    for op in sorted(conteo):
        principal = max(sorted(conteo[op]), key=lambda k: sum(conteo[op][k]))
        n0, n1 = conteo[op][principal]
        if n0 + n1 >= min_comentarios:
            anterior = votos.get(principal, [0, 0])
            votos[principal] = [anterior[0] + n0, anterior[1] + n1]
    return votos
