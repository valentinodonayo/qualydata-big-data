"""Entrena el clasificador de ruido de RESPALDO con la verdad del set de PRÁCTICA.

Desde el bloque de "vecinos en hilos", el ruido se decide mirando si los comentarios
más parecidos participan de hilos (pipeline/narrativas.py, paso 5), sin entrenar nada.
Este clasificador queda solo como respaldo para un set que casi no tenga hilos.

Uso (se corre una vez; el resultado queda en modelos/ruido.json y va al repo):
    uv run python entrenar_ruido.py

El README del desafío permite entrenar con la verdad de práctica. Para que sirva
con los temas nuevos del set final, el clasificador solo usa rasgos que no dependen
del tema (largo, parecido con la narrativa más cercana, margen, si es pregunta).

Antes de guardar, se valida "por tema": se entrena con 2 temas y se prueba con el
tercero, que el modelo nunca vio. Si eso anda bien, es buena señal para el final.
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from pipeline import datos, embeddings, narrativas

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datos", type=Path, default=Path("datos/practica"), help="Set con carpeta verdad/")
    ap.add_argument("--salida", type=Path, default=Path("modelos/ruido.json"))
    ap.add_argument("--cache", type=Path, default=Path("cache"))
    args = ap.parse_args()
    DATOS, CACHE, SALIDA = args.datos, args.cache, args.salida
    d = datos.cargar(DATOS, CACHE)
    com = d["comentarios"]
    emb, indice = embeddings.calcular(com["texto"].tolist(), CACHE)
    final, pos_padre = narrativas.detectar(com, emb, indice, CACHE, solo_sin_filtrar=True)
    pos, rasgos, cercana = narrativas.rasgos_ruido(final, com, emb, indice, pos_padre)

    verdad = pd.read_csv(DATOS / "verdad/comentarios.csv.gz", dtype=str, keep_default_na=False)
    narr_real = com["comentario_id"].map(verdad.set_index("comentario_id")["narrativa_id"]).to_numpy()
    y = narr_real[pos] == "ninguna"
    X = rasgos[narrativas.RASGOS_RUIDO].to_numpy()
    print(f"Raíces dudosas: {len(pos)} (ruido real: {y.mean():.1%})")

    # Validación por tema: el "tema" de cada dudosa = tema de su narrativa más cercana,
    # que sacamos de la verdad (primera letra del id: L, B, H)
    tema_de = {}
    for k in set(cercana):
        reales = pd.Series(narr_real[(final == k)])
        reales = reales[reales != "ninguna"]
        tema_de[k] = reales.str[0].value_counts().index[0] if len(reales) else "?"
    tema = np.array([tema_de[k] for k in cercana])
    for t in sorted(set(tema) - {"?"}):
        if len(set(tema) - {"?"}) < 2:
            break   # con un solo tema no hay contra qué validar
        modelo = LogisticRegression(max_iter=1000).fit(X[tema != t], y[tema != t])
        print(f"  entrenado SIN el tema {t} -> F1 de ruido en el tema {t}: "
              f"{f1_score(y[tema == t], modelo.predict(X[tema == t])):.3f}")

    # Modelo final: con todos los datos de práctica
    modelo = LogisticRegression(max_iter=1000).fit(X, y)
    guardado = {"rasgos": narrativas.RASGOS_RUIDO,
                "coeficientes": dict(zip(narrativas.RASGOS_RUIDO, modelo.coef_[0].round(6).tolist())),
                "intercepto": float(modelo.intercept_[0]),
                "entrenado_con": "verdad del set de práctica (raíces dudosas)",
                "f1_practica": round(f1_score(y, modelo.predict(X)), 4)}
    SALIDA.parent.mkdir(exist_ok=True)
    SALIDA.write_text(json.dumps(guardado, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Guardado en {SALIDA}: {guardado}")


if __name__ == "__main__":
    main()
