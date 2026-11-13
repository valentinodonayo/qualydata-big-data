"""Tests de la señal local de polaridad (qué bando "refuta") y de cómo entra en la regla final."""
import numpy as np
import pandas as pd

from pipeline import ia, postura


def test_fraccion_refutacion_cuenta_textos_distintos():
    textos = ["Eso es MENTIRA, ya lo desmintieron!!", "eso es mentira ya lo desmintieron",   # el mismo texto
              "Nos están robando en la cara", "Dejen de inventar"]
    # 3 textos distintos, 2 con refutación
    assert abs(postura.fraccion_refutacion(textos) - 2 / 3) < 1e-9
    assert postura.fraccion_refutacion([]) == 0.0


def test_voto_local_se_abstiene_si_es_debil_o_se_contradice():
    assert postura.voto_local(0.02, 0.001) is None        # ninguna medida supera su umbral
    assert postura.voto_local(0.30, 0.0) == 1             # el bando 0 refuta más -> apoya el 1
    assert postura.voto_local(-0.30, -0.02) == 0          # las dos dicen que refuta el bando 1
    assert postura.voto_local(0.0, 0.02) == 1             # alcanza con una si la otra no opina
    assert postura.voto_local(0.30, -0.02) is None        # se contradicen: no vota


def test_senal_local_con_datos_de_juguete():
    # Bando 0 denuncia, bando 1 desmiente. Embeddings de 2 números: el 2º es "cuánto refuta".
    com = pd.DataFrame({"texto": ["nos roban la plata", "la obra está parada", "eso es mentira", "dejen de inventar"]})
    emb = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    extra = {"narr_num": np.array(["n01"] * 4, dtype=object)}
    bando = np.array([0, 0, 1, 1])
    r = postura.senal_local(com, emb, np.arange(4), extra, bando, direccion=np.array([0.0, 1.0]))
    assert r["n01"]["voto"] == 0          # refuta el bando 1 -> apoya el 0
    assert r["n01"]["lexico"] == -1.0


def test_combinar_usa_la_senal_local_solo_para_desempatar():
    # Sin IA y bandos parejos: gana la señal local aunque señale al bando más chico
    assert ia._combinar([], [500, 510], None, local=0)[0] == 0
    # Sin señal local (se abstuvo): la regla de siempre, el bando más grande
    assert ia._combinar([], [500, 510], None, local=None)[0] == 1
    # Margen grande (>= 54%): el tamaño le gana a la señal local
    assert ia._combinar([], [400, 600], None, local=0)[0] == 1
    # Operación coordinada en el otro bando: le gana a la señal local
    assert ia._combinar([], [500, 510], [0, 100], local=0)[0] == 1
    # IA unánime: la señal local no cambia nada
    assert ia._combinar([1, 1, 1], [510, 500], None, local=0) == ia._combinar([1, 1, 1], [510, 500], None)
