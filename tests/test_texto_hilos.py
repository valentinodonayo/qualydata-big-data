"""Tests de normalización de texto y de hilos de respuestas."""
from pipeline import hilos
from pipeline import texto as tx


def test_para_embeddings_saca_menciones_links_y_letras_repetidas():
    t = "HAY QUEEEE DECIRLO!! @majo_ok87 mirá https://x.co #ElLitioEsFuturo 💯"
    assert tx.para_embeddings(t) == "hay que decirlo!! mirá ellitioesfuturo"


def test_para_comparar_saca_acentos_y_signos():
    assert tx.para_comparar("¡Qué DESASTRE, señor!") == "que desastre señor"


def test_texto_vacio_o_nulo_no_rompe():
    assert tx.para_embeddings(None) == ""
    assert tx.para_comparar("") == ""


def test_respuesta_corta():
    assert hilos.es_corta("Tal cual")
    assert not hilos.es_corta("No es cierto, la obra sigue avanzando según el informe")


def test_raices_con_padre_inexistente_y_ciclo():
    ids = ["a", "b", "c", "d", "x", "y"]
    padres = ["", "a", "b", "zzz", "y", "x"]   # d responde a algo que no existe; x<->y ciclo
    padres = [p or None for p in padres]
    r = hilos.raices(ids, padres)
    assert r["b"] == "a" and r["c"] == "a"
    assert r["d"] == "d"                       # padre inexistente -> es raíz
    assert r["x"] in {"x", "y"} and r["x"] == r["y"]   # el ciclo se corta sin colgarse


def test_posturas_por_hilo_corta_hereda_larga_invierte():
    ids = ["r", "c1", "l1", "c2"]
    padres = [None, "r", "r", "l1"]
    textos = ["raíz", "Exacto", "Para nada, eso es mentira y lo saben todos", "Tal cual"]
    p = hilos.posturas_por_hilo(ids, padres, textos, {"r": "apoya"})
    assert p == {"r": "apoya", "c1": "apoya", "l1": "rechaza", "c2": "rechaza"}
