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
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(ia, "_cargar_clave", lambda *a: False)


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


# ---------------- Gemini (sin internet: se reemplaza el POST por respuestas armadas) ----------------

def _respuesta_gemini(datos):
    import json
    return 200, {"candidates": [{"content": {"parts": [{"text": json.dumps(datos)}]}}],
                 "usageMetadata": {"promptTokenCount": 800, "candidatesTokenCount": 90, "thoughtsTokenCount": 10}}


def test_elige_gemini_primero(monkeypatch):
    monkeypatch.setattr(ia, "_cargar_clave", lambda nombre="": nombre == "GEMINI_API_KEY")
    assert ia._elegir_proveedor() == "gemini"
    monkeypatch.setattr(ia, "_cargar_clave", lambda nombre="": nombre == "ANTHROPIC_API_KEY")
    assert ia._elegir_proveedor() == "claude"
    monkeypatch.setattr(ia, "_cargar_clave", lambda nombre="": False)
    assert ia._elegir_proveedor() is None


def test_gemini_responde_guarda_en_cache_y_reproduce_sin_clave(monkeypatch, tmp_path):
    ruta = tmp_path / "c.json"
    pedidos = []

    def post_falso(cuerpo):
        pedidos.append(cuerpo)
        if len(pedidos) == 1:          # la primera forma del pedido "no la acepta": prueba la siguiente
            return 400, "campo desconocido"
        return _respuesta_gemini(RESPUESTA)

    monkeypatch.setattr(ia, "_cargar_clave", lambda nombre="": nombre == "GEMINI_API_KEY")
    monkeypatch.setattr(ia, "_post_gemini", post_falso)
    crono = costo.Cronometro()
    apoya, titulos = ia.decidir(RESUMEN, crono, ruta_cache=ruta)
    assert apoya == {"n01": 1} and titulos["n01"]["titulo"] == "El dique trae agua"
    assert len(pedidos) == 2 and crono.tokens_ia == 900 and crono.usd_ia == 0   # capa gratis
    assert crono.ia_modelo == ia.MODELO_GEMINI

    # Ahora SIN clave (como el jurado): la caché alcanza y da lo mismo, sin llamar a nadie
    monkeypatch.setattr(ia, "_cargar_clave", lambda *a: False)
    monkeypatch.setattr(ia, "_post_gemini", lambda c: pytest.fail("no debería llamar a la API"))
    apoya2, titulos2 = ia.decidir(RESUMEN, costo.Cronometro(), ruta_cache=ruta)
    assert apoya2 == apoya and titulos2 == titulos


def test_gemini_error_usa_respaldo(monkeypatch, tmp_path):
    monkeypatch.setattr(ia, "_cargar_clave", lambda nombre="": nombre == "GEMINI_API_KEY")
    monkeypatch.setattr(ia, "_post_gemini", lambda c: (403, "clave inválida"))
    apoya, titulos = ia.decidir(RESUMEN, costo.Cronometro(), ruta_cache=tmp_path / "c.json")
    assert apoya == {"n01": 0} and titulos == {}            # respaldo: el bando más grande
    assert not (tmp_path / "c.json").exists()               # nada que guardar


def test_clave_de_ejemplo_no_cuenta(monkeypatch, tmp_path):
    """El texto 'poner-tu-clave-aca' de .env.example no es una clave."""
    import importlib
    monkeypatch.undo()
    importlib.reload(ia)
    monkeypatch.setenv("GEMINI_API_KEY", "poner-tu-clave-aca")
    assert ia._cargar_clave("GEMINI_API_KEY") is False
