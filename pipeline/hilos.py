"""Herramientas para los hilos de respuestas (columna respuesta_a).

Lo que medimos en los datos de práctica (y coincide con las reglas del desafío):
- El 100% de las respuestas tiene la MISMA narrativa que el comentario al que responde.
- Las respuestas CORTAS (≤ 4 palabras: "Tal cual", "Exacto") acompañan:
  misma postura que el padre (99% de los casos).
- Las respuestas LARGAS contradicen: postura OPUESTA al padre (99,8% de los casos).

Entonces: si sabemos la narrativa y la postura de la "raíz" de un hilo,
podemos deducir las de todas sus respuestas, gratis y sin IA.
"""
from pipeline import texto as tx

PALABRAS_RESPUESTA_CORTA = 4
OPUESTA = {"apoya": "rechaza", "rechaza": "apoya"}


def es_corta(texto: str) -> bool:
    """True si la respuesta es corta (acompaña al padre)."""
    return len(tx.para_comparar(texto).split()) <= PALABRAS_RESPUESTA_CORTA


def raices(ids, padres):
    """Para cada comentario, devuelve el id de la raíz de su hilo.

    ids:    lista de comentario_id
    padres: lista con el respuesta_a de cada uno (None si no responde a nadie)

    Una respuesta a un comentario que no existe en los datos se toma como raíz.
    Si hubiera un ciclo (A responde a B y B a A), se corta y se toma como raíz.
    """
    padre_de = {i: p for i, p in zip(ids, padres) if p is not None and p == p}  # p == p descarta NaN
    existentes = set(ids)
    raiz = {}

    for inicio in ids:
        camino, actual = [], inicio
        visitados = set()
        # Subimos padre por padre hasta llegar a algo sin padre (o ya resuelto)
        while actual not in raiz:
            camino.append(actual)
            visitados.add(actual)
            p = padre_de.get(actual)
            if p is None or p not in existentes or p in visitados:
                raiz[actual] = actual       # este es la raíz
                break
            actual = p
        r = raiz[actual]
        for c in camino:                    # todos los del camino comparten esa raíz
            raiz[c] = r
    return raiz


def posturas_por_hilo(ids, padres, textos, postura_raiz):
    """Completa la postura de cada respuesta a partir de la de su padre.

    postura_raiz: dict {comentario_id: "apoya"/"rechaza"} con la postura de las raíces.
    Recorre de arriba hacia abajo: corta -> igual que el padre; larga -> la opuesta.
    """
    padre_de = {i: p for i, p in zip(ids, padres) if p is not None and p == p}
    texto_de = dict(zip(ids, textos))
    postura = dict(postura_raiz)

    def resolver(c, profundidad=0):
        if c in postura:
            return postura[c]
        p = padre_de.get(c)
        if p is None or profundidad > 500:      # sin padre conocido o hilo absurdo
            return ""
        pp = resolver(p, profundidad + 1)
        if not pp:
            postura[c] = ""
        else:
            postura[c] = pp if es_corta(texto_de.get(c, "")) else OPUESTA[pp]
        return postura[c]

    for c in ids:
        resolver(c)
    return postura
