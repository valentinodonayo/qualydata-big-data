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


def test_adhesion_por_hashtag_propio():
    """Una cuenta de poca actividad que usa el hashtag de la operación se suma a ella;
    una que usa un hashtag popular entre la gente común, no."""
    df = _armar()
    extra = [
        _fila(9001, "p50", "rezagada", BASE + 9 * 86400, "Vamos que se puede #VamosCardonal"),
        _fila(9002, "p51", "comun", BASE + 9 * 86400, "Qué lindo día #Primavera"),
    ]
    # #Primavera lo usan muchas cuentas genuinas (no es "propio" de ninguna operación)
    extra += [_fila(9100 + g, f"p{60 + g}", f"vecino{g}", BASE + g * 3600, "Llegó #Primavera")
              for g in range(15)]
    grupo, resumen = coordinacion.detectar(pd.concat([df, pd.DataFrame(extra)], ignore_index=True))
    assert grupo.get("rezagada") == grupo["bot0"]
    assert "comun" not in grupo and not any(c.startswith("vecino") for c in grupo)
    assert "adhesión" in resumen[0]["capas"]


# ---------------- Capa G: llegadas raras ----------------

def test_llegadas_raras_suma_la_cuenta_de_cobertura_y_no_la_muy_activa():
    """'cobertura' llega con 2 miembros en momentos tranquilos de 4 publicaciones -> se suma.
    'activa' coincide con miembros en 4 publicaciones, pero siempre en momentos con mucha gente
    y comentando en 200 publicaciones -> su puntaje es bajo y su densidad mínima: no se suma."""
    filas, i = [], 0
    grupo = {"m1": 0, "m2": 0}
    for k in range(4):
        pub = f"q{k}"
        t0 = BASE + k * 86400
        for cuenta, dt in (("m1", 0), ("m2", 60), ("cobertura", 120)):
            i += 1
            filas.append(_fila(i, pub, cuenta, t0 + dt, "texto"))
        # momento concurrido en otra publicación: miembros + 'activa' + 25 personas
        pub2 = f"r{k}"
        for cuenta in ["m1", "activa"] + [f"gente{g}" for g in range(25)]:
            i += 1
            filas.append(_fila(i, pub2, cuenta, t0 + 7200, "texto"))
    for g in range(200):              # 'activa' comenta en muchísimas publicaciones
        i += 1
        filas.append(_fila(i, f"s{g}", "activa", BASE + g * 3600 + 30 * 86400, "texto"))
    nuevas = coordinacion._llegadas_raras(pd.DataFrame(filas), grupo)
    assert nuevas == {"cobertura": 0}
