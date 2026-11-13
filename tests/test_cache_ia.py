"""Tests de la caché de IA: misma pregunta -> misma huella; cualquier cambio -> otra huella."""
from pipeline import cache_ia


def test_huella_estable_y_sensible_a_cambios():
    h = cache_ia.huella("m", {"a": 1, "b": 2}, "sis", "msj")
    assert h == cache_ia.huella("m", {"b": 2, "a": 1}, "sis", "msj")      # el orden no importa
    assert h != cache_ia.huella("m2", {"a": 1, "b": 2}, "sis", "msj")     # otro modelo
    assert h != cache_ia.huella("m", {"a": 1, "b": 3}, "sis", "msj")      # otros parámetros
    assert h != cache_ia.huella("m", {"a": 1, "b": 2}, "sis", "msj!")     # otro mensaje


def test_guardar_escribir_y_releer(tmp_path):
    ruta = tmp_path / "cache.json"
    c = cache_ia.CacheIA(ruta)
    assert c.buscar("x") is None
    c.escribir()
    assert not ruta.exists()                      # sin respuestas nuevas no escribe nada
    c.guardar("x", {"respuesta": {"ok": True}, "tokens_entrada": 10, "tokens_salida": 2, "usd": 0.1})
    c.escribir()
    c2 = cache_ia.CacheIA(ruta)
    assert c2.buscar("x")["respuesta"] == {"ok": True}
    assert c2.aciertos == 1


def test_cache_rota_no_rompe(tmp_path):
    ruta = tmp_path / "cache.json"
    ruta.write_text("{esto no es json")
    assert cache_ia.CacheIA(ruta).buscar("x") is None
