"""Tests del proveedor "gateway" (Vercel AI Gateway) SIN internet: el POST se reemplaza por
respuestas armadas. Cubren: elección de proveedor, reintentos, costo de lista, las dos órdenes
de los bandos (sesgo de posición), la regla que combina señales y la reproducción sin clave."""
import json

import numpy as np
import pandas as pd
import pytest

from pipeline import costo, ia, postura

EJ_0 = [f"el dique trae agua para todos, ejemplo {i}" for i in range(12)]
EJ_1 = [f"mentira, el dique no trae agua, ejemplo {i}" for i in range(12)]
RESUMEN = {"n01": {"tamano": [90, 100], "ejemplos": [EJ_0, EJ_1]}}     # el bando 1 es el más grande
MODELOS = ["proveedor/modelo-a", "proveedor/modelo-b"]
CLAVE_FALSA = "clave-de-mentira-para-tests"


@pytest.fixture(autouse=True)
def gateway_de_prueba(monkeypatch):
    """Hay clave de gateway (falsa), sin esperas, y dos modelos inventados."""
    for nombre in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(nombre, raising=False)
    monkeypatch.setenv("AI_GATEWAY_API_KEY", CLAVE_FALSA)
    monkeypatch.setattr(ia, "_cargar_clave", lambda nombre="": nombre == "AI_GATEWAY_API_KEY")
    monkeypatch.setattr(ia, "MODELOS_GATEWAY", MODELOS)
    monkeypatch.setattr(ia, "ESPERA_REINTENTO", 0)
    monkeypatch.setattr(ia, "PEDIDOS_POR_MINUTO", 0)


def _ok(bando, entrada=1000, salida=100):
    """Respuesta 200 con la forma de la API tipo OpenAI."""
    datos = {"afirmacion": "El dique trae agua", "bando_que_afirma": bando,
             "titulo": f"El dique trae agua ({bando})", "frases_clave": ["dique", "agua"]}
    return 200, {"choices": [{"message": {"content": json.dumps(datos)}}],
                 "usage": {"prompt_tokens": entrada, "completion_tokens": salida}}


def _modelo_que_entiende(cuerpo):
    """Modelo "bueno": elige el bando que afirma (el que NO dice 'mentira'), esté primero o segundo."""
    mensaje = cuerpo["messages"][1]["content"]
    bloque_a = mensaje.split("BANDO B:")[0]
    return _ok("B" if "mentira" in bloque_a else "A")


def test_elige_gateway_primero(monkeypatch):
    monkeypatch.setattr(ia, "_cargar_clave", lambda nombre="": True)      # hay las tres claves
    assert ia._elegir_proveedor() == "gateway"


def test_pedido_bien_armado_y_costo_de_lista(monkeypatch):
    pedidos = []
    monkeypatch.setattr(ia, "_post_gateway", lambda cuerpo: (pedidos.append(cuerpo), _ok("A"))[1])
    entrada = ia._preguntar_gateway("n01", "BANDO A:\n- x\n\nBANDO B:\n- y", "openai/gpt-5-mini")
    cuerpo = pedidos[0]
    assert cuerpo["model"] == "openai/gpt-5-mini"
    assert cuerpo["response_format"]["json_schema"]["schema"] == ia.ESQUEMA      # salida JSON según ESQUEMA
    assert cuerpo["messages"][0]["content"] == ia.PROMPTS[ia.PROMPT_ACTUAL]
    assert entrada["respuesta"]["bando_que_afirma"] == "A"
    # Costo = tokens × precio de lista del modelo (aunque la capa sea gratis)
    precio = ia.PRECIOS_GATEWAY["openai/gpt-5-mini"]
    assert entrada["usd"] == pytest.approx(1000 * precio[0] + 100 * precio[1])
    assert CLAVE_FALSA not in json.dumps(entrada) and CLAVE_FALSA not in json.dumps(cuerpo)


def test_reintenta_ante_429_y_5xx(monkeypatch):
    respuestas = [(429, "muchos pedidos"), (503, "ocupado"), _ok("B")]
    llamadas = []
    monkeypatch.setattr(ia, "_post_gateway", lambda c: (llamadas.append(1), respuestas[len(llamadas) - 1])[1])
    entrada = ia._preguntar_gateway("n01", "msj", MODELOS[0])
    assert len(llamadas) == 3 and entrada["respuesta"]["bando_que_afirma"] == "B"


def test_no_reintenta_un_403_y_devuelve_none(monkeypatch):
    llamadas = []
    monkeypatch.setattr(ia, "_post_gateway", lambda c: (llamadas.append(1), (403, "Free tier users do not..."))[1])
    assert ia._preguntar_gateway("n01", "msj", MODELOS[0]) is None
    assert len(llamadas) == 1


@pytest.mark.parametrize("contenido", ["", "no soy json", '{"bando_que_afirma": "C", "titulo": "x", "frases_clave": []}',
                                       '{"titulo": "falta el bando"}', None])
def test_respuesta_inservible_devuelve_none(monkeypatch, contenido):
    monkeypatch.setattr(ia, "_post_gateway", lambda c: (200, {"choices": [{"message": {"content": contenido}}]}))
    assert ia._preguntar_gateway("n01", "msj", MODELOS[0]) is None


def test_sin_internet_usa_el_respaldo_sin_romper(monkeypatch, tmp_path):
    def sin_red(cuerpo):
        raise OSError("sin internet")
    monkeypatch.setattr(ia, "_post_gateway", sin_red)
    crono = costo.Cronometro()
    apoya, titulos = ia.decidir(RESUMEN, crono, ruta_cache=tmp_path / "c.json")
    assert apoya == {"n01": 1} and titulos == {}          # respaldo: el bando más grande
    assert crono.usd_ia == 0 and not (tmp_path / "c.json").exists()


def test_vota_con_las_dos_ordenes_guarda_y_reproduce_sin_clave(monkeypatch, tmp_path):
    ruta = tmp_path / "c.json"
    pedidos = []
    monkeypatch.setattr(ia, "_post_gateway", lambda c: (pedidos.append(c), _modelo_que_entiende(c))[1])
    crono = costo.Cronometro()
    apoya, titulos = ia.decidir(RESUMEN, crono, ruta_cache=ruta, huella_datos="set1")
    # 2 modelos × 2 órdenes = 4 pedidos; todos dicen que afirma el bando 0 aunque sea el más chico
    assert len(pedidos) == 4 and apoya == {"n01": 0}
    assert crono.ia_detalle["n01"]["motivo"].startswith("IA unánime")
    assert crono.tokens_ia == 4 * 1100 and crono.ia_modelo == " + ".join(MODELOS)
    assert CLAVE_FALSA not in ruta.read_text(encoding="utf-8")      # la caché nunca guarda la clave
    assert "el dique trae agua para todos" not in ruta.read_text(encoding="utf-8")   # ni los comentarios

    # Ahora SIN clave (como el jurado): la caché alcanza, da lo mismo y no llama a nadie
    monkeypatch.setattr(ia, "_cargar_clave", lambda *a: False)
    monkeypatch.setattr(ia, "_post_gateway", lambda c: pytest.fail("no debería llamar a la API"))
    crono2 = costo.Cronometro()
    apoya2, titulos2 = ia.decidir(RESUMEN, crono2, ruta_cache=ruta, huella_datos="set1")
    assert (apoya2, titulos2) == (apoya, titulos) and crono2.tokens_ia == crono.tokens_ia

    # Si el pipeline cambió un poco (otro ejemplo, bandos numerados al revés): se reconoce igual
    movido = {"n07": {"tamano": [100, 90], "ejemplos": [EJ_1, EJ_0[:11] + ["texto nuevo"]]}}
    apoya3, _ = ia.decidir(movido, costo.Cronometro(), ruta_cache=ruta, huella_datos="set1")
    assert apoya3 == {"n07": 1}                             # EJ_0 ahora es el bando 1: ese apoya


def test_modelo_que_elige_por_posicion_no_vota(monkeypatch, tmp_path):
    """Un modelo que siempre dice "A" se contradice al invertir los bandos: se lo ignora."""
    monkeypatch.setattr(ia, "_post_gateway", lambda c: _ok("A"))
    crono = costo.Cronometro()
    apoya, titulos = ia.decidir(RESUMEN, crono, ruta_cache=tmp_path / "c.json")
    assert apoya == {"n01": 1}                              # sin votos válidos: el bando más grande
    assert crono.ia_detalle["n01"]["contradicciones"] == 2 and crono.ia_detalle["n01"]["votos"] == []
    assert "n01" in titulos                                 # el título de la IA igual se aprovecha


def test_votos_por_modelo():
    votos, contradicciones = ia.votos_por_modelo([("a", 1), ("a", 1), ("b", 0), ("b", 1), ("c", 0)])
    assert votos == [1, 0] and contradicciones == 1         # "a" vota 1, "b" se contradice, "c" vota 0


def test_combinar_senales():
    parejo, desparejo = [100, 104], [100, 150]
    # Sin IA: el más grande; con operación coordinada clara, el bando de la operación
    assert ia._combinar([], parejo)[0] == 1
    assert ia._combinar([], parejo, coordinadas=[50, 2])[0] == 0
    # IA unánime: gana la IA aunque sea el bando chico
    assert ia._combinar([0, 0, 0], parejo)[0] == 0
    # IA dividida: decide la operación coordinada, y si no hay, la mayoría de los modelos
    assert ia._combinar([0, 0, 1], parejo, coordinadas=[1, 60])[0] == 1
    assert ia._combinar([0, 0, 1], parejo)[0] == 0
    # IA empatada: el bando más grande
    assert ia._combinar([0, 1], desparejo)[0] == 1
    # Operación repartida entre los dos bandos (no es clara): no se usa
    assert ia._combinar([], parejo, coordinadas=[30, 20])[0] == 1


def test_ejemplos_variados_y_bando_de_operaciones():
    # 2 micro-grupos en el bando: debe salir un ejemplo de cada uno (no dos copias de la misma frase)
    textos = ["la obra del hospital está parada hace muchos meses ya"] * 3 + \
             ["la obra del hospital está PARADA hace muchos meses yaaa"] + \
             ["se robaron toda la plata del hospital nuevo, es una vergüenza"] * 2
    micro = np.array([0, 0, 0, 0, 1, 1])
    emb = np.eye(6, 4)[[0, 0, 0, 0, 1, 1]] + 0.0
    variados = postura._ejemplos_variados(np.arange(6), micro, textos, emb, np.arange(6))
    assert variados[0].startswith("la obra") and variados[1].startswith("se robaron")
    assert len(variados) == 2                                # las casi-copias no se repiten

    com = pd.DataFrame({"cuenta_id": ["c1", "c1", "c2", "c3", "c3", "c3"]})
    extra = {"narr_num": np.array(["n01"] * 6, dtype=object)}
    bando = np.array([1, 1, 1, 0, 1, -1])
    votos = postura.bando_de_operaciones(com, bando, extra, {"c1": "op01", "c2": "op01"}, min_comentarios=3)
    assert votos == {"n01": [0, 3]}                          # c3 no es coordinada: no cuenta
