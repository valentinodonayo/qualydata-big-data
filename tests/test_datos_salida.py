"""Tests de lectura defensiva de datos sucios y del formato de entrega."""
import gzip
import json

import pandas as pd

from pipeline import costo, datos, salida, validar


def _escribir_gz(ruta, texto):
    with gzip.open(ruta, "wt", encoding="utf-8") as f:
        f.write(texto)


def test_cargar_datos_sucios_no_se_cae(tmp_path):
    carpeta = tmp_path / "set"
    carpeta.mkdir()
    # Columnas desordenadas, una columna extra, un id repetido, una fecha rota,
    # un texto vacío, una fila sin id, una cuenta vacía y una respuesta a un comentario inexistente
    _escribir_gz(carpeta / "comentarios.csv.gz", (
        "texto,comentario_id,extra,publicacion_id,cuenta_id,fecha,respuesta_a,me_gusta\n"
        "hola mundo,c1,x,p1,u1,2026-10-05T10:00:00-03:00,,3\n"
        "repetido,c1,x,p1,u1,2026-10-05T10:00:00-03:00,,3\n"
        "fecha rota,c2,x,p1,u2,ayer a la tarde,,no-es-numero\n"
        ",c3,x,p1,u3,2026-10-05T11:00:00Z,c1,0\n"
        "sin id,,x,p1,u4,2026-10-05T11:00:00-03:00,,0\n"
        "sin cuenta,c4,x,p1,,2026-10-05T12:00:00-03:00,c999,0\n"
    ))
    # cuentas.csv sin la columna "siguiendo"; publicaciones sin comprimir; falta fuentes.csv
    _escribir_gz(carpeta / "cuentas.csv.gz", "cuenta_id,usuario,fecha_creacion,seguidores\nu1,ana,2020-01-01,10\n")
    (carpeta / "publicaciones.csv").write_text("publicacion_id,fuente_id,fecha,titular\np1,f1,2026-10-05T09:00:00-03:00,T\n")

    d = datos.cargar(carpeta, tmp_path / "cache")
    com, cal = d["comentarios"], d["calidad"]
    assert list(com["comentario_id"]) == ["c1", "c2", "c3", "c4"]     # sin duplicado ni fila sin id
    assert com.loc[com.comentario_id == "c2", "ts"].isna().all()       # fecha rota -> vacía
    assert com.loc[com.comentario_id == "c3", "ts"].notna().all()      # fecha en UTC se entiende
    assert com.loc[com.comentario_id == "c4", "cuenta_id"].item() == "desconocida"
    assert cal["comentarios_fecha_invalida"] == 1
    assert cal["comentarios_texto_vacio"] == 1
    assert cal["respuestas_a_comentario_inexistente"] == 1
    assert cal["cuentas_columnas_faltantes"] == ["siguiendo"]
    assert cal["fuentes_archivo_faltante"] is True
    # Las cuentas que comentan sin metadata igual aparecen
    assert set(datos.todas_las_cuentas(d)) == {"u1", "u2", "u3"}


def test_entrega_vacia_es_valida(tmp_path):
    """Aunque todas las etapas fallen, la entrega tiene que cumplir el formato."""
    salida.escribir_entrega(tmp_path, ["c1", "c2"], ["u1"])
    costo.escribir_costo(tmp_path, costo.Cronometro(), 2, "DuckDB", "prueba")
    assert validar.validar_entrega(tmp_path, ["c1", "c2"], ["u1"]) == []


def test_entrega_completa_y_reglas_de_formato(tmp_path):
    salida.escribir_entrega(
        tmp_path, ["c1", "c2", "c3"], ["u1", "u2"],
        narrativa={"c1": "n01", "c2": "n02"},
        postura={"c1": "APOYA", "c2": "cualquiera", "c3": "apoya"},   # c3 es ruido
        grupo={"u2": "op01"},
        narrativas=[{"narrativa": "n01", "titulo": "t", "frases_clave": "a|b",
                     "fecha_origen": salida.ts_a_iso(1790000000)}])
    costo.escribir_costo(tmp_path, costo.Cronometro(), 3, "DuckDB", "prueba")
    com = pd.read_csv(tmp_path / "comentarios.csv", dtype=str, keep_default_na=False)
    assert com.set_index("comentario_id")["postura"].to_dict() == {"c1": "apoya", "c2": "", "c3": ""}
    narr = pd.read_csv(tmp_path / "narrativas.csv", dtype=str, keep_default_na=False)
    assert set(narr["narrativa"]) == {"n01", "n02"}     # n02 se agrega aunque no venga en la tabla
    assert narr.loc[narr.narrativa == "n01", "fecha_origen"].item().endswith("-03:00")
    problemas = validar.validar_entrega(tmp_path, ["c1", "c2", "c3"], ["u1", "u2"])
    # n02 se agregó sin fecha: el validador lo avisa (la fecha es parte del formato)
    assert problemas == ["fecha_origen inválida en n02: ''"]
    assert json.loads((tmp_path / "costo.json").read_text())["comentarios_procesados"] == 3


def test_validador_detecta_errores(tmp_path):
    (tmp_path / "comentarios.csv").write_text("comentario_id,narrativa,postura\nc1,,apoya\nc1,n01,quizas\n")
    problemas = validar.validar_entrega(tmp_path)
    assert any("repetidos" in p for p in problemas)
    assert any("falta cuentas.csv" in p for p in problemas)
    assert any("falta costo.json" in p for p in problemas)
