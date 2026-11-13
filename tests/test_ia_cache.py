"""Tests de la IA con caché, SIN llamar al modelo (no hace falta clave ni internet)."""
import pytest

from pipeline import cache_ia, costo, ia

EJ_0 = [f"el dique trae agua para todos, ejemplo {i}" for i in range(12)]
EJ_1 = [f"mentira, el dique no trae agua, ejemplo {i}" for i in range(12)]
RESUMEN = {"n01": {"tamano": [100, 90], "ejemplos": [EJ_0, EJ_1]}}
RESPUESTA = {"afirmacion": "El dique trae agua", "bando_que_afirma": "B",
             "titulo": "El dique trae agua", "frases_clave": ["dique", "agua"]}


@pytest.fixture(autouse=True)
def sin_clave(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(ia, "_cargar_clave", lambda: False)


def _entrada(**extra):
    return {"respuesta": RESPUESTA, "tokens_entrada": 900, "tokens_salida": 100, "usd": 0.00014, **extra}


def test_sin_cache_ni_clave_usa_el_respaldo(tmp_path):
    crono = costo.Cronometro()
    apoya, titulos = ia.decidir(RESUMEN, crono, ruta_cache=tmp_path / "c.json")
    assert apoya == {"n01": 0} and titulos == {}          # el bando más grande (0) apoya
    assert crono.usd_ia == 0


def test_acierto_exacto_reproduce_sin_clave(tmp_path):
    ruta = tmp_path / "c.json"
    c = cache_ia.CacheIA(ruta)
    clave = cache_ia.huella(ia.MODELO, ia.PARAMETROS, ia.INSTRUCCIONES, ia._mensaje(RESUMEN["n01"]))
    c.guardar(clave, _entrada())
    c.escribir()
    crono = costo.Cronometro()
    apoya, titulos = ia.decidir(RESUMEN, crono, ruta_cache=ruta)
    assert apoya == {"n01": 1}                             # la IA dijo que afirma el bando B (= 1)
    assert titulos["n01"]["titulo"] == "El dique trae agua"
    assert crono.tokens_ia == 1000 and crono.usd_ia == pytest.approx(0.00014)   # costo de la corrida original


def test_acierto_parecido_con_bandos_invertidos(tmp_path):
    """Otra máquina numeró los bandos al revés y cambió un ejemplo: igual se reconoce."""
    ruta = tmp_path / "c.json"
    c = cache_ia.CacheIA(ruta)
    c.guardar("otra-huella", _entrada(etiqueta="n01", huella_datos="set1", ejemplos=[EJ_1, EJ_0]))
    c.escribir()
    distinto = {"n01": {"tamano": [100, 90], "ejemplos": [EJ_0[:11] + ["texto nuevo"], EJ_1]}}
    apoya, _ = ia.decidir(distinto, costo.Cronometro(), ruta_cache=ruta, huella_datos="set1")
    # Guardado: afirma B = EJ_0. Acá EJ_0 es el bando 0 -> apoya el bando 0
    assert apoya == {"n01": 0}
    # Con otro set de datos no se reutiliza (respaldo: el más grande, que también es 0 -> probamos con tamaños)
    distinto["n01"]["tamano"] = [10, 90]
    apoya, _ = ia.decidir(distinto, costo.Cronometro(), ruta_cache=ruta, huella_datos="otro-set")
    assert apoya == {"n01": 1}
