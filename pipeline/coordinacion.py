"""Etapa 5: detectar cuentas coordinadas (bots o "granjas") y agruparlas en operaciones.

Idea general: una cuenta sola no es sospechosa; lo sospechoso es un GRUPO de cuentas
que se comporta junto más de lo que el azar explicaría. Buscamos 4 tipos de señal
("capas") y conectamos las cuentas que las comparten:

  A) Copiar y pegar: el mismo texto (largo) publicado por 3 a 80 cuentas en <= 30 min.
  B) Llegar juntas: comentar en la misma publicación en la misma franja de 15 min,
     en momentos con poca gente (si hay mucha gente, coincidir no dice nada).
  C) Responderse: una cuenta le responde a la otra.
     -> A, B y C se suman por PAR de cuentas. Un par queda conectado si coincide en
        >= 3 publicaciones distintas y eso es >= 30% de la actividad de la menos activa.
        Los grupos = cuentas conectadas entre sí (5 o más).

  E) Adhesión (después de armar las operaciones): se suma a una operación la cuenta que
     usa su HASHTAG PROPIO (uno que en >= 2/3 de sus usos aparece en cuentas de esa
     operación) o que intercambia >= 3 respuestas con >= 2 miembros distintos. Rescata
     cuentas de poca actividad que no llegaron a 3 coincidencias con un mismo par.

  G) Llegadas raras (después de E): se suma a una operación la cuenta que llega a las mismas
     publicaciones que sus miembros en momentos TRANQUILOS. Cada coincidencia en una franja de
     15 min pesa 1/(cuentas presentes - 1): coincidir con pocas pesa mucho, en un momento viral
     casi nada. Se suma si el puntaje hacia la operación es >= 1,4, si ese puntaje es >= 0,03 por
     publicación en la que comenta (densidad: una cuenta muy activa junta puntaje por volumen, no
     por coordinación) y si coincidió (en franjas de 60 min) en >= 2 publicaciones distintas.
     Se repite hasta 3 veces: una cuenta recién sumada puede delatar a otra. Rescata cuentas de "cobertura" con textos variados.

  H) Tanda de creación (al final): si casi todos los miembros de una operación se crearon en
     pocos días (5% a 95% de las fechas en <= 10 días) y con la misma FORMA de usuario (letras y
     cantidad de dígitos, ej. "palabra+4 dígitos", en >= 80% de los miembros), se suma la cuenta
     no marcada creada en esa ventana (+-1 día) con esa misma forma. Si encaja en dos operaciones,
     va a la que comparte más publicaciones con ella. Cuentas sin fecha o sin usuario: se ignoran.

  D) Red de respuestas "con ciclos": en una charla normal las respuestas forman un
     árbol (cada uno contesta a uno). Un grupo coordinado se responde entre todos y
     forma una red llena de ciclos. Con Leiden buscamos comunidades en el grafo de
     "quién responde a quién" y marcamos las que tienen >= 1,6 conexiones por cuenta
     (un árbol tiene casi exactamente 1). Esta capa atrapa a las campañas de textos
     variados, que A y B no ven.

Todo es conteo y grafos: no usa IA y tarda pocos segundos con 200.000 comentarios.
"""
import itertools
import logging
import os
import re
import unicodedata
from collections import defaultdict

import igraph as ig
import leidenalg as la
import pandas as pd

# --- Parámetros (medidos con el set de práctica; ver experimentos/runs.csv) ---------
LARGO_MINIMO_TEXTO = 25        # capa A: textos más cortos ("jaja", "gracias") se repiten solos
CUENTAS_MIN_TEXTO, CUENTAS_MAX_TEXTO = 3, 80   # capa A: cuántas cuentas repiten el texto
VENTANA_TEXTO_S = 30 * 60      # capa A: copias dentro de 30 minutos
FRANJA_S = 15 * 60             # capa B: tamaño de la franja de tiempo
MAX_CUENTAS_FRANJA = 30        # capa B: momentos con más gente no cuentan
MIN_PUBLICACIONES = 3          # par: coincidir en al menos 3 publicaciones distintas
MIN_FRACCION = 0.3             # par: ...y que sea >= 30% de la actividad de la menos activa
FRACCION_PROPIA_UN_VINCULO = 0.1   # par único que es < 10% de la actividad de la cuenta: se suelta
MIN_CUENTAS_GRUPO = 5          # un grupo necesita 5 cuentas o más
RESOLUCION_RESPUESTAS = 0.05   # capa D: resolución de Leiden (CPM): comunidades chicas y densas
MIN_CONEXIONES_POR_CUENTA = 1.6  # capa D: un árbol tiene ~1; los grupos coordinados, 2 o más
HASHTAG_MIN_USOS = 10          # capa E: un hashtag cuenta si se usó al menos 10 veces...
HASHTAG_FRACCION_OP = 2 / 3    # ...y una mayoría clara (2 de cada 3 usos) es de UNA operación
MIN_RESPUESTAS_ADHESION = 3    # capa E: respuestas intercambiadas con miembros de la operación
MIN_MIEMBROS_ADHESION = 2      # ...con al menos 2 miembros distintos
RAREZA_MINIMA = 1.4           # capa G: suma de pesos 1/(presentes-1) hacia la operación
DENSIDAD_RAREZA = 0.03         # capa G: ...y ese puntaje dividido las publicaciones de la cuenta
MIN_PUBLICACIONES_RAREZA = 2    # capa G: publicaciones distintas en común (franjas de 60 min)
PASADAS_RAREZA = 3              # capa G: se repite con las operaciones ya agrandadas (tope)
FRANJA_LARGA_S = 60 * 60       # capa G: franja para contar publicaciones distintas en común
TANDA_MAX_DIAS = 10            # capa H: ancho máximo de la ventana de creación (percentiles 5-95)
TANDA_FORMA_MIN = 0.8          # capa H: fracción de miembros con la forma de usuario dominante
TANDA_MIN_MIEMBROS = 5         # capa H: miembros con metadata para evaluar una tanda
SEMILLA = int(os.environ.get("QUALY_SEMILLA", "42"))   # fija; se cambia solo para medir estabilidad

CUENTA_DESCONOCIDA = "desconocida"   # la pone datos.py cuando falta la cuenta: no se mezcla


def _normalizar(texto):
    """Texto "esqueleto" para comparar copias: sin mayúsculas, acentos, hashtags,
    menciones, links, signos ni letras estiradas ("siiiii" -> "si")."""
    t = unicodedata.normalize("NFD", str(texto).lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"#\w+|@\w+|https?:\S+", " ", t)
    t = re.sub(r"[^a-zñ0-9 ]", " ", t)
    t = re.sub(r"(\w)\1{2,}", r"\1", t)
    return re.sub(r"\s+", " ", t).strip()


def _pares_por_senales(com):
    """Capas A, B y C: para cada par de cuentas, en qué publicaciones coincidieron."""
    pares = defaultdict(set)   # (cuenta_a, cuenta_b) -> {publicaciones}

    def sumar(a, b, publicacion):
        if a != b:
            pares[(a, b) if a < b else (b, a)].add(publicacion)

    con_fecha = com[com["ts"].notna()]
    ts = con_fecha["ts"].astype("int64")

    # Capa A: copiar y pegar
    norm = con_fecha["texto"].map(_normalizar)
    largos = con_fecha.assign(norm=norm, t=ts)[norm.str.len() >= LARGO_MINIMO_TEXTO]
    for _, g in largos.groupby("norm"):
        if not CUENTAS_MIN_TEXTO <= g["cuenta_id"].nunique() <= CUENTAS_MAX_TEXTO:
            continue
        filas = sorted(zip(g["t"], g["cuenta_id"], g["publicacion_id"]))
        for i in range(len(filas)):
            for j in range(i + 1, len(filas)):
                if filas[j][0] - filas[i][0] > VENTANA_TEXTO_S:
                    break   # están ordenadas por tiempo: las siguientes quedan más lejos
                sumar(filas[i][1], filas[j][1], filas[i][2])

    # Capa B: llegar juntas. Usamos dos "grillas" de franjas corridas media franja,
    # para no perder a dos cuentas que caen justo a cada lado de un corte.
    for corrimiento in (0, FRANJA_S // 2):
        franja = (ts + corrimiento) // FRANJA_S
        clave = con_fecha["publicacion_id"] + "@" + franja.astype(str)
        for k, cuentas in con_fecha.groupby(clave)["cuenta_id"]:
            presentes = sorted(set(cuentas))
            if len(presentes) < 2 or len(presentes) > MAX_CUENTAS_FRANJA:
                continue
            publicacion = k.split("@")[0]
            for a, b in itertools.combinations(presentes, 2):
                sumar(a, b, publicacion)

    # Capa C: responderse
    autor = dict(zip(com["comentario_id"], com["cuenta_id"]))
    for cuenta, publicacion, padre in zip(com["cuenta_id"], com["publicacion_id"], com["respuesta_a"]):
        otra = autor.get(padre) if isinstance(padre, str) else None
        if otra:
            sumar(cuenta, otra, publicacion)
    return pares


def _grupos_por_pares(com, pares):
    """Une las cuentas de los pares fuertes y devuelve los grupos de 5 o más."""
    publicaciones_de = com.groupby("cuenta_id")["publicacion_id"].nunique().to_dict()
    vecinos = defaultdict(set)
    for (a, b), pubs in pares.items():
        if len(pubs) < MIN_PUBLICACIONES:
            continue
        if len(pubs) / min(publicaciones_de[a], publicaciones_de[b]) < MIN_FRACCION:
            continue   # coinciden poco comparado con todo lo que comentan: casualidad
        vecinos[a].add(b)
        vecinos[b].add(a)
    # Una cuenta MUY activa que solo se conecta por UN par, y ese par es una parte mínima de lo que
    # comenta (< FRACCION_PROPIA_UN_VINCULO), coincide por volumen, no por coordinación: se suelta.
    for c in sorted(vecinos):
        if len(vecinos[c]) == 1:
            (otra,) = vecinos[c]
            par = (c, otra) if c < otra else (otra, c)
            if len(pares[par]) / publicaciones_de[c] < FRACCION_PROPIA_UN_VINCULO:
                vecinos[c].discard(otra)
                vecinos[otra].discard(c)

    # Componentes conexas: recorremos la red y juntamos a todos los que se tocan
    grupos, visto = [], set()
    for inicio in list(vecinos):
        if inicio in visto:
            continue
        pila, componente = [inicio], []
        while pila:
            c = pila.pop()
            if c in visto:
                continue
            visto.add(c)
            componente.append(c)
            pila.extend(vecinos[c] - visto)
        if len(componente) >= MIN_CUENTAS_GRUPO:
            grupos.append(set(componente))
    return grupos


def _grupos_por_red_de_respuestas(com):
    """Capa D: comunidades del grafo de respuestas que tienen muchos más ciclos que un árbol."""
    autor = dict(zip(com["comentario_id"], com["cuenta_id"]))
    peso = defaultdict(int)   # (a, b) -> cuántas veces se respondieron
    for cuenta, padre in zip(com["cuenta_id"], com["respuesta_a"]):
        otra = autor.get(padre) if isinstance(padre, str) else None
        if otra and otra != cuenta and CUENTA_DESCONOCIDA not in (cuenta, otra):
            peso[(cuenta, otra) if cuenta < otra else (otra, cuenta)] += 1
    if not peso:
        return []

    nombres = sorted({c for par in peso for c in par})
    numero = {n: i for i, n in enumerate(nombres)}
    grafo = ig.Graph(n=len(nombres), edges=[(numero[a], numero[b]) for a, b in peso],
                     edge_attrs={"weight": list(peso.values())})
    # CPM busca comunidades más densas que la "resolución"; con 0,05 salen grupos chicos
    particion = la.find_partition(grafo, la.CPMVertexPartition, weights="weight",
                                  resolution_parameter=RESOLUCION_RESPUESTAS, seed=SEMILLA)
    grupos = []
    for comunidad in particion:
        if len(comunidad) < MIN_CUENTAS_GRUPO:
            continue
        conexiones = grafo.subgraph(comunidad).ecount()
        if conexiones / len(comunidad) >= MIN_CONEXIONES_POR_CUENTA:
            grupos.append({nombres[i] for i in comunidad})
            logging.info("Coordinación (capa D): comunidad de %s cuentas con %s conexiones",
                         len(comunidad), conexiones)
    return grupos


def _normalizar_hashtag(tag):
    """'#DiqueSííí' -> '#diquesi' (sin mayúsculas, acentos ni letras estiradas)."""
    t = unicodedata.normalize("NFD", tag.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"(\w)\1{2,}", r"\1", t)


def _adhesion(com, grupo):
    """Capa E: cuentas que se suman a una operación ya detectada. Devuelve {cuenta: operación}."""
    nuevas = {}
    # E1) Hashtag propio de una operación
    usos = defaultdict(list)   # hashtag -> [cuenta, cuenta, ...] (una entrada por uso)
    for cuenta, texto in zip(com["cuenta_id"], com["texto"]):
        for tag in re.findall(r"#\w+", str(texto)):
            usos[_normalizar_hashtag(tag)].append(cuenta)
    for tag, cuentas in usos.items():
        if len(cuentas) < HASHTAG_MIN_USOS:
            continue
        por_op = pd.Series([grupo.get(c, "") for c in cuentas])
        por_op = por_op[por_op != ""].value_counts()
        if por_op.empty or por_op.iloc[0] / len(cuentas) < HASHTAG_FRACCION_OP:
            continue
        for c in set(cuentas):
            if c not in grupo:
                nuevas.setdefault(c, por_op.index[0])
    # E2) Muchas respuestas con varios miembros de la misma operación
    autor = dict(zip(com["comentario_id"], com["cuenta_id"]))
    respuestas = defaultdict(lambda: defaultdict(list))   # cuenta -> operación -> [miembros]
    for cuenta, padre in zip(com["cuenta_id"], com["respuesta_a"]):
        otra = autor.get(padre) if isinstance(padre, str) else None
        if not otra or otra == cuenta:
            continue
        if otra in grupo and cuenta not in grupo:
            respuestas[cuenta][grupo[otra]].append(otra)
        if cuenta in grupo and otra not in grupo:
            respuestas[otra][grupo[cuenta]].append(cuenta)
    for c, por_op in sorted(respuestas.items()):
        for op, miembros in sorted(por_op.items()):
            if len(miembros) >= MIN_RESPUESTAS_ADHESION and len(set(miembros)) >= MIN_MIEMBROS_ADHESION:
                nuevas.setdefault(c, op)
    nuevas.pop(CUENTA_DESCONOCIDA, None)
    if nuevas:
        logging.info("Coordinación (capa E): %s cuentas se suman por hashtag propio o respuestas", len(nuevas))
    return nuevas


def _llegadas_raras(com, grupo):
    """Capa G: puntaje de coincidencias en momentos tranquilos con miembros de cada operación.
    Devuelve {cuenta: operación} para las cuentas no miembro que superan los umbrales."""
    con_fecha = com[com["ts"].notna()]
    ts = con_fecha["ts"].astype("int64")
    puntaje = defaultdict(lambda: defaultdict(float))    # cuenta -> operación -> puntaje
    publicaciones = defaultdict(lambda: defaultdict(set))  # cuenta -> operación -> {publicaciones}
    for ancho in (FRANJA_S, FRANJA_LARGA_S):
        for corrimiento in (0, ancho // 2):              # dos grillas corridas, como en la capa B
            clave = con_fecha["publicacion_id"] + "@" + ((ts + corrimiento) // ancho).astype(str)
            for k, cuentas in con_fecha.groupby(clave)["cuenta_id"]:
                presentes = set(cuentas)
                miembros = defaultdict(int)
                for c in presentes:
                    if c in grupo:
                        miembros[grupo[c]] += 1
                if len(presentes) < 2 or not miembros:
                    continue
                peso = 1.0 / (len(presentes) - 1)
                publicacion = k.split("@")[0]
                for c in presentes:
                    if c in grupo:
                        continue
                    for op, n in miembros.items():
                        if ancho == FRANJA_S:
                            puntaje[c][op] += peso * n / 2   # /2: cada momento se cuenta en dos grillas
                        else:
                            publicaciones[c][op].add(publicacion)
    actividad = com.groupby("cuenta_id")["publicacion_id"].nunique().to_dict()
    nuevas = {}
    for c in sorted(puntaje):
        candidatas = [(p, op) for op, p in puntaje[c].items()
                      if p >= RAREZA_MINIMA and p / max(1, actividad.get(c, 1)) >= DENSIDAD_RAREZA
                      and len(publicaciones[c][op]) >= MIN_PUBLICACIONES_RAREZA]
        if candidatas:
            nuevas[c] = max(candidatas, key=lambda x: (x[0], -x[1]))[1]
    nuevas.pop(CUENTA_DESCONOCIDA, None)
    if nuevas:
        logging.info("Coordinación (capa G): %s cuentas se suman por llegar en momentos tranquilos", len(nuevas))
    return nuevas


def _forma_usuario(usuario):
    """'trabajoorgullo8302' -> 'a9999': letras = 'a', cada dígito = '9', el resto igual."""
    u = re.sub(r"\d", "9", str(usuario).lower())
    return re.sub(r"[a-zñáéíóú]+", "a", u)


def _tanda_de_creacion(com, cuentas, grupo):
    """Capa H: cuentas creadas en la misma tanda (fechas y forma de usuario) que una operación."""
    meta = cuentas[["cuenta_id", "usuario", "fecha_creacion"]].copy()
    meta["fecha"] = pd.to_datetime(meta["fecha_creacion"], errors="coerce")
    meta = meta[meta["fecha"].notna() & meta["usuario"].fillna("").astype(str).str.strip().ne("")]
    meta = meta.drop_duplicates("cuenta_id")
    if meta.empty:
        return {}
    meta["forma"] = meta["usuario"].map(_forma_usuario)
    meta["op"] = meta["cuenta_id"].map(grupo)
    tandas = {}
    for op, m in meta[meta["op"].notna()].groupby("op"):
        if len(m) < TANDA_MIN_MIEMBROS:
            continue
        desde, hasta = m["fecha"].quantile(0.05), m["fecha"].quantile(0.95)
        formas = m["forma"].value_counts()
        if (hasta - desde).days <= TANDA_MAX_DIAS and formas.iloc[0] / len(m) >= TANDA_FORMA_MIN:
            tandas[op] = (desde - pd.Timedelta(days=1), hasta + pd.Timedelta(days=1), formas.index[0])
    if not tandas:
        return {}
    pubs_de = com.groupby("cuenta_id")["publicacion_id"].agg(set).to_dict()
    pubs_op = {op: set().union(*[pubs_de.get(c, set()) for c, g in grupo.items() if g == op]) for op in tandas}
    nuevas = {}
    for c, fecha, forma in zip(meta["cuenta_id"], meta["fecha"], meta["forma"]):
        if c in grupo or c == CUENTA_DESCONOCIDA:
            continue
        candidatas = [op for op, (desde, hasta, f) in tandas.items() if desde <= fecha <= hasta and forma == f]
        if candidatas:
            mias = pubs_de.get(c, set())
            # int(): la columna "op" queda como número decimal porque tiene vacíos (NaN)
            nuevas[c] = int(min(candidatas, key=lambda op: (-len(mias & pubs_op[op]), op)))
    if nuevas:
        logging.info("Coordinación (capa H): %s cuentas se suman por haberse creado en la tanda de una operación",
                     len(nuevas))
    return nuevas


def detectar(com: pd.DataFrame, metadata_cuentas: pd.DataFrame = None):
    """Devuelve (grupo, resumen).

    grupo:   {cuenta_id: "op01", ...} solo para las cuentas marcadas como coordinadas
    resumen: lista de operaciones con sus cuentas y por qué capa se detectaron
    """
    com = com[com["cuenta_id"] != CUENTA_DESCONOCIDA]
    grupos_abc = _grupos_por_pares(com, _pares_por_senales(com))
    grupos_d = _grupos_por_red_de_respuestas(com)

    # Juntamos las dos listas. Si un grupo de la capa D se pisa con uno de A-C,
    # se fusionan (es la misma operación vista desde dos señales).
    operaciones = [(g, {"texto/tiempo/respuestas"}) for g in grupos_abc]
    for g in grupos_d:
        for cuentas, capas in operaciones:
            if cuentas & g:
                cuentas |= g
                capas.add("red de respuestas")
                break
        else:
            operaciones.append((set(g), {"red de respuestas"}))

    # Capa E: adhesión de cuentas a las operaciones ya armadas
    provisorio = {c: i for i, (cuentas, _) in enumerate(operaciones) for c in sorted(cuentas)}
    for c, i in _adhesion(com, provisorio).items():
        operaciones[i][0].add(c)
        operaciones[i][1].add("adhesión")

    # Capa G: llegadas raras (se calcula con las operaciones ya completadas por E)
    for _ in range(PASADAS_RAREZA):
        provisorio = {c: i for i, (cuentas, _) in enumerate(operaciones) for c in sorted(cuentas)}
        nuevas = _llegadas_raras(com, provisorio)
        if not nuevas:
            break
        for c, i in nuevas.items():
            operaciones[i][0].add(c)
            operaciones[i][1].add("llegadas raras")

    # Capa H: tanda de creación (solo si hay metadata de cuentas)
    if metadata_cuentas is not None and not metadata_cuentas.empty:
        provisorio = {c: i for i, (cs, _) in enumerate(operaciones) for c in sorted(cs)}
        for c, i in _tanda_de_creacion(com, metadata_cuentas, provisorio).items():
            operaciones[i][0].add(c)
            operaciones[i][1].add("tanda de creación")

    # Etiquetas op01, op02... de la operación más grande a la más chica
    operaciones.sort(key=lambda x: -len(x[0]))
    grupo, resumen = {}, []
    for i, (cuentas, capas) in enumerate(operaciones, start=1):
        etiqueta = f"op{i:02d}"
        for c in cuentas:
            grupo.setdefault(c, etiqueta)
        resumen.append({"grupo": etiqueta, "cuentas": len(cuentas), "capas": sorted(capas)})
    logging.info("Coordinación: %s operaciones, %s cuentas marcadas", len(resumen), len(grupo))
    return grupo, resumen
