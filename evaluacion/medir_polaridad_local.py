"""Mide la señal LOCAL de polaridad (sin IA): ¿acierta qué bando "apoya" cada narrativa?

La señal está en pipeline/postura.py (senal_local): el bando que más "refuta" (desmiente,
relativiza, reta al otro) es el que rechaza. Acá se arman los bandos igual que el pipeline y se
compara el voto de la señal contra la respuesta correcta:
- en práctica, contra la verdad (datos/practica/verdad);
- en otro set (p. ej. el final, que no tiene verdad), contra una entrega ya hecha (--entrega):
  el bando que en esa entrega quedó como "apoya".

Uso:
    python evaluacion/medir_polaridad_local.py --datos datos/practica
    python evaluacion/medir_polaridad_local.py --datos datos/final --entrega entrega

No llama a ninguna API ni usa claves. Tarda ~1 minuto por set (con los embeddings en caché).
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import numpy as np            # noqa: E402
import pandas as pd           # noqa: E402

from pipeline import ia, postura   # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datos", type=Path, default=RAIZ / "datos/practica")
    ap.add_argument("--cache", type=Path, default=RAIZ / "cache")
    ap.add_argument("--entrega", type=Path, default=None,
                    help="carpeta con comentarios.csv de una entrega: se usa su postura como respuesta correcta")
    args = ap.parse_args()

    # Mismas etapas que correr.py hasta tener los dos bandos de cada narrativa
    import correr
    estado = {}
    args.modelo_ruido = RAIZ / "modelos" / "ruido.json"
    correr.etapa_datos(estado, args)
    correr.etapa_embeddings(estado, args)
    correr.etapa_narrativas(estado, args)
    com = estado["datos"]["comentarios"]
    extra = estado["extra_narrativas"]
    bando_com, resumen = postura.calcular_bandos(com, estado["emb"], estado["emb_indice"], extra)
    direccion = postura.direccion_refutacion(args.cache)
    local = postura.senal_local(com, estado["emb"], estado["emb_indice"], extra, bando_com, direccion)

    # Respuesta correcta por comentario: de la verdad (práctica) o de una entrega ya hecha
    if args.entrega:
        ref = pd.read_csv(args.entrega / "comentarios.csv", dtype=str, keep_default_na=False)
        col_narr = "narrativa"
    else:
        ref = pd.read_csv(args.datos / "verdad" / "comentarios.csv.gz", dtype=str, keep_default_na=False)
        col_narr = "narrativa_id"
    ref = ref.set_index("comentario_id").reindex(com["comentario_id"].to_numpy())
    narr = extra["narr_num"]

    cuenta = Counter()
    print("narr  real  bandos(raíces)  %mayor  léxico  prototipos  voto local  tamaño  regla sin IA")
    for k in sorted(resumen):
        en = np.where((narr == k) & (bando_com >= 0))[0]
        real = Counter(x for x in ref[col_narr].to_numpy()[en] if x).most_common(1)[0][0]
        apoyo = [float((ref["postura"].to_numpy()[en][bando_com[en] == b] == "apoya").mean()) for b in (0, 1)]
        correcto = int(apoyo[1] > apoyo[0])                     # el bit que hay que acertar
        tam = resumen[k]["tamano"]
        mayor = 0 if tam[0] >= tam[1] else 1
        parejo = max(tam) / max(sum(tam), 1) < ia.MARGEN_GRANDE
        v = local[k]
        final, _ = ia._combinar([], tam, None, v["voto"])       # lo que decide el pipeline con --sin-ia
        marca = lambda x: "se abstiene" if x is None else ("OK " if x == correcto else "MAL")
        print(f"{k}  {real:4s}  {str(tam):15s} {max(tam) / max(sum(tam), 1):.3f}  {v['lexico']:+.3f}  {v['prototipos']:+.4f}"
              f"     {marca(v['voto']):11s} {marca(mayor)}     {marca(final)}" + ("   (bandos parejos)" if parejo else ""))
        for nombre, x in (("local", v["voto"]), ("tamano", mayor), ("regla", final)):
            cuenta[nombre, marca(x).strip()] += 1
            if parejo:
                cuenta[nombre + "_parejos", marca(x).strip()] += 1
    n = len(resumen)
    for nombre, titulo in (("local", "Señal local sola"), ("tamano", "Bando más grande"), ("regla", "Regla sin IA (tamaño + local)")):
        print(f"{titulo}: {cuenta[nombre, 'OK']} aciertos, {cuenta[nombre, 'MAL']} errores, "
              f"{cuenta[nombre, 'se abstiene']} abstenciones de {n}  |  en bandos parejos (< {ia.MARGEN_GRANDE:.0%}): "
              f"{cuenta[nombre + '_parejos', 'OK']} aciertos, {cuenta[nombre + '_parejos', 'MAL']} errores, "
              f"{cuenta[nombre + '_parejos', 'se abstiene']} abstenciones")


if __name__ == "__main__":
    main()
