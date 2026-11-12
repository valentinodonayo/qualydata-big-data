"""Tests de coordinación con casos armados a mano: una operación y dos trampas."""
import pandas as pd

from pipeline import coordinacion

BASE = 1_790_000_000   # una fecha cualquiera, en segundos


def _fila(i, pub, cuenta, ts, texto, resp=None):
    return {"comentario_id": f"c{i}", "publicacion_id": pub, "cuenta_id": cuenta,
            "ts": ts, "texto": texto, "respuesta_a": resp, "me_gusta": 0}


def _armar():
    filas, i = [], 0
    # Operación: 6 cuentas pegan el mismo texto, en 4 publicaciones distintas, en ráfagas de minutos
    for k, pub in enumerate(["p1", "p2", "p3", "p4"]):
        for b in range(6):
            i += 1
            filas.append(_fila(i, pub, f"bot{b}", BASE + k * 86400 + b * 60,
                               "El festival deja millones para todos los comercios del pueblo #VamosCardonal"))
    # Trampa 1, momento viral: 200 cuentas distintas comentan UNA publicación en 10 minutos
    for g in range(200):
        i += 1
        filas.append(_fila(i, "p9", f"gente{g}", BASE + 3 * 3600 + g * 3, f"opinión número {g} sobre el tema"))
    # Trampa 2, frase copiada sin coordinación: 10 cuentas, mismo texto, separadas por días
    for g in range(10):
        i += 1
        filas.append(_fila(i, f"p{20 + g}", f"copion{g}", BASE + g * 2 * 86400,
                           "No hay que creer todo lo que se dice en las redes sociales"))
    return pd.DataFrame(filas)


def test_detecta_la_operacion_y_no_las_trampas():
    grupo, resumen = coordinacion.detectar(_armar())
    marcadas = set(grupo)
    assert {f"bot{b}" for b in range(6)} <= marcadas
    assert len({grupo[f"bot{b}"] for b in range(6)}) == 1          # todas en la MISMA operación
    assert not any(c.startswith(("gente", "copion")) for c in marcadas)
    assert len(resumen) == 1


def test_sin_respuestas_ni_coincidencias_no_marca_a_nadie():
    df = _armar()
    df = df[~df["cuenta_id"].str.startswith("bot")]
    grupo, _ = coordinacion.detectar(df)
    assert grupo == {}
