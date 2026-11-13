"""Pruebas de la señal de ruido: "¿los comentarios parecidos participan de hilos?"

Armamos a mano un set mínimo con dos tipos de texto:
- tipo A (narrativa): algunos de sus comentarios reciben respuestas,
- tipo B (ruido): ninguno recibe respuestas.
Un comentario suelto de tipo A NO debe ser ruido; uno de tipo B sí.
"""
import numpy as np
import pandas as pd

from pipeline import narrativas


def _set_de_juguete():
    # 4 textos distintos: 0 y 1 son casi iguales (tipo A); 2 y 3 son casi iguales (tipo B)
    emb = np.array([[1.0, 0.0], [0.99, 0.14], [0.0, 1.0], [0.14, 0.99]])
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    # 40 copias del texto 0, 20 del 1, 40 del 2 y 20 del 3: para juntar los 50 vecinos que se
    # miran, cada texto necesita sumar a su "casi igual" (ninguno llega solo a 50 copias)
    emb_indice = np.repeat([0, 1, 2, 3], [40, 20, 40, 20])
    con_contenido = np.ones(120, dtype=bool)
    en_hilo = np.zeros(120, dtype=bool)
    en_hilo[:12] = True          # solo comentarios del texto 0 (tipo A) están en hilos
    return emb, emb_indice, con_contenido, en_hilo


def test_vecinos_de_narrativa_participan_de_hilos_y_los_de_ruido_no():
    emb, emb_indice, con_contenido, en_hilo = _set_de_juguete()
    sueltos = np.where(~en_hilo)[0]
    tasa = narrativas.tasa_hilos_vecinos(emb, emb_indice, con_contenido, en_hilo, sueltos)
    tipo_a = emb_indice[sueltos] <= 1
    assert (tasa[tipo_a] > 0.05).all()       # sus copias o frases vecinas sí están en hilos
    assert (tasa[~tipo_a] == 0).all()        # nadie parecido recibe respuestas -> ruido


def test_separar_ruido_marca_solo_los_sueltos_sin_vecinos_en_hilos():
    emb, emb_indice, con_contenido, en_hilo = _set_de_juguete()
    n = len(emb_indice)
    # Los primeros 12 comentarios (tipo A) reciben una respuesta cada uno: sumamos 12 respuestas largas
    emb_indice = np.concatenate([emb_indice, np.full(12, 1)])
    pos_padre = np.concatenate([np.full(n, -1), np.arange(12)])
    con_contenido = np.ones(n + 12, dtype=bool)
    pos_raiz = np.concatenate([np.arange(n), np.arange(12)])
    com = pd.DataFrame({"texto": ["una frase cualquiera de prueba"] * (n + 12)})
    # Antes del paso 5: los que están en hilos ya tienen narrativa 0; el resto, sin decidir (-1),
    # salvo el ruido del texto 3, que se había "colado" en la narrativa 0
    final = np.full(n + 12, -1)
    final[:12] = 0
    final[n:] = 0
    final[emb_indice == 3] = 0
    micro_com = emb_indice.copy()      # en el juguete, cada texto distinto es su propio micro-grupo
    nuevo = narrativas._separar_ruido(final, com, emb, emb_indice, pos_padre, con_contenido, pos_raiz,
                                      micro_com, None)
    assert (nuevo[emb_indice <= 1] == 0).all()     # tipo A: todo rescatado a la narrativa
    assert (nuevo[emb_indice >= 2] == -1).all()    # tipo B: todo ruido, incluso el que se había colado


def test_rescate_prefiere_la_narrativa_de_las_copias_en_hilos():
    # Texto 0 = narrativa 0, texto 2 = narrativa 1. El texto 1 es una "consigna" que el agrupamiento
    # dejó sin micro-grupo y que se PARECE más al texto 2... pero sus copias con respuestas
    # quedaron (por el hilo) en la narrativa 0. Las copias sueltas tienen que ir a la 0.
    emb = np.array([[1.0, 0.0], [0.3, 0.95], [0.0, 1.0]])
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    raices = np.repeat([0, 1, 2], [30, 20, 30])                  # texto de cada comentario raíz
    con_respuesta = np.concatenate([np.arange(0, 10), np.arange(30, 35), np.arange(50, 60)])
    texto_resp = np.where(raices[con_respuesta] == 2, 2, 0)      # las respuestas usan el texto 0 o el 2
    emb_indice = np.concatenate([raices, texto_resp])
    n, total = len(raices), len(raices) + len(con_respuesta)
    pos_padre = np.concatenate([np.full(n, -1), con_respuesta])
    pos_raiz = np.concatenate([np.arange(n), con_respuesta])
    micro_com = np.where(emb_indice == 0, 0, np.where(emb_indice == 2, 1, -1))
    final = np.where(emb_indice == 0, 0, np.where(emb_indice == 2, 1, -1))
    final[30:35] = 0                                             # copias de la consigna que están en hilos
    com = pd.DataFrame({"texto": ["una frase cualquiera de prueba"] * total})
    nuevo = narrativas._separar_ruido(final, com, emb, emb_indice, pos_padre, np.ones(total, dtype=bool),
                                      pos_raiz, micro_com, None)
    assert (nuevo[35:50] == 0).all()        # sueltas de la consigna: narrativa de sus copias, no la "más parecida"
    assert (nuevo[emb_indice == 2] == 1).all() and (nuevo[emb_indice == 0] == 0).all()
