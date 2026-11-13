"""Herramientas para los hilos de respuestas (columna respuesta_a).

Lo que medimos en los datos de práctica (y coincide con las reglas del desafío):
- El 100% de las respuestas tiene la MISMA narrativa que el comentario al que responde.
- Las respuestas CORTAS (≤ 4 palabras: "Tal cual", "Exacto") acompañan:
  misma postura que el padre (99% de los casos).
- Las respuestas LARGAS contradicen: postura OPUESTA al padre (99,8% de los casos).

Entonces: si sabemos la narrativa y la postura de la "raíz" de un hilo,
podemos deducir las de todas sus respuestas, gratis y sin IA.

Dos excepciones medidas sobre la verdad de práctica (ver experimentos):
- Un 1% de las respuestas cortas CONTRADICE al padre: son frases con contenido
  propio ("los turnos tardan meses") que también circulan como raíces. Si el
  mismo texto aparece como raíz con una postura clara, esa es la postura.
- Un 0,2% de las respuestas largas ACUERDA con el padre: casi todas empiezan con
  una fórmula de acuerdo ("Eso, ...", "Tal cual, ..."). Si la respuesta abre con
  una de esas fórmulas (y no la niega), acompaña al padre aunque sea larga.
"""
import re
from collections import Counter, defaultdict

from pipeline import texto as tx

PALABRAS_RESPUESTA_CORTA = 4
OPUESTA = {"apoya": "rechaza", "rechaza": "apoya"}

# Fórmulas de acuerdo al inicio de la respuesta (texto ya normalizado). El lookahead
# descarta que la fórmula venga negada ("eso no", "claro que no", "obvio que no").
_ACUERDO = re.compile(
    r"^(tal cual(?:ito)?|eso|eso mismo|exacto|exactamente|asi es(?: nomas)?|idem|"
    r"de acuerdo|totalmente(?: de acuerdo)?|muy cierto|obvio|claro(?: que si)?|"
    r"por supuesto|mas claro)\b(?!\s*(que\s+)?no\b)")
MIN_USOS_TEXTO_RAIZ = 2        # el texto tiene que aparecer como raíz al menos 2 veces
FRACCION_TEXTO_RAIZ = 0.7      # ...y con una mayoría clara de una sola postura
MIN_PALABRAS_TEXTO_RAIZ = 3    # textos de 1-2 palabras ("jaja", "+1") no llevan postura propia


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
    Recorre de arriba hacia abajo. Orden de decisiones para cada respuesta:
    1. Si abre con una fórmula de acuerdo ("eso, ...", "tal cual, ..."), acompaña
       al padre aunque sea larga: la fórmula afirma lo que dice el padre.
    2. Si su texto normalizado también aparece como raíz con una postura clara
       (>= MIN_USOS_TEXTO_RAIZ usos y >= FRACCION_TEXTO_RAIZ de una sola postura),
       vale esa postura: una copia exacta afirma lo mismo como respuesta que como
       raíz, aunque le responda a alguien del otro bando.
    3. Si es corta, acompaña al padre.
    4. Si no, contradice al padre.
    """
    padre_de = {i: p for i, p in zip(ids, padres) if p is not None and p == p}
    texto_de = dict(zip(ids, textos))
    postura = dict(postura_raiz)

    # Postura de cada texto cuando aparece como raíz (las raíces ya tienen bando)
    votos_raiz = defaultdict(Counter)
    for c, p in postura_raiz.items():
        votos_raiz[tx.para_comparar(texto_de.get(c, ""))][p] += 1

    def postura_del_texto(norm, palabras):
        if palabras < MIN_PALABRAS_TEXTO_RAIZ:
            return ""
        v = votos_raiz.get(norm)
        if not v or sum(v.values()) < MIN_USOS_TEXTO_RAIZ:
            return ""
        p, k = v.most_common(1)[0]
        return p if k / sum(v.values()) >= FRACCION_TEXTO_RAIZ else ""

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
            norm = tx.para_comparar(texto_de.get(c, ""))
            palabras = len(norm.split())
            propia = postura_del_texto(norm, palabras)
            if _ACUERDO.search(norm):
                postura[c] = pp
            elif propia:
                postura[c] = propia
            elif palabras <= PALABRAS_RESPUESTA_CORTA:
                postura[c] = pp
            else:
                postura[c] = OPUESTA[pp]
        return postura[c]

    for c in ids:
        resolver(c)
    return postura
