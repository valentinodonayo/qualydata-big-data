"""Mide cuántas POLARIDADES acierta la IA en práctica (el "bit" por narrativa: qué bando apoya).

Por qué existe: correr todo el pipeline para probar un prompt tarda y mezcla cosas. Acá se arma
UNA vez el `resumen` de bandos igual que el pipeline (mismas etapas, mismas funciones) y se guarda
en cache/polaridad_<set>.json junto con la respuesta correcta según la verdad. Después se puede
preguntar a distintos modelos / prompts muchas veces y contar aciertos en segundos.

Uso:
    # 1) armar el resumen (una vez por set de datos; ~1 minuto)
    python evaluacion/medir_polaridad.py armar --datos datos/practica
    # 2) medir un modelo, 3 repeticiones, con el prompt "v2"
    python evaluacion/medir_polaridad.py medir --modelo openai/gpt-5-mini --prompt v2 --repeticiones 3
    # 3) ver qué decidiría la regla final (IA + tamaño + operación coordinada) con las respuestas
    #    ya guardadas por `medir`, sin llamar a nadie
    python evaluacion/medir_polaridad.py regla --modelos openai/gpt-5-mini,openai/gpt-4.1 --prompt v3

`medir` pregunta cada narrativa DOS veces (bandos en orden y al revés) y respeta el límite de
5 pedidos por minuto por modelo de la capa gratis: 13 narrativas tardan ~5 minutos por modelo.
Las respuestas de `medir` se guardan en cache/polaridad_respuestas_<modelo>.json (NO es la caché del
pipeline: es solo para no pagar dos veces el mismo experimento).
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import numpy as np            # noqa: E402
import pandas as pd           # noqa: E402

from pipeline import cache_ia, ia, postura   # noqa: E402


def armar(carpeta_datos: Path, carpeta_cache: Path) -> dict:
    """Corre datos + embeddings + narrativas + bandos (igual que correr.py) y devuelve
    {narrativa: {tamano, ejemplos, antiguos, coordinadas, [verdad...]}}."""
    import correr
    from pipeline import coordinacion
    args = argparse.Namespace(datos=carpeta_datos, cache=carpeta_cache,
                              modelo_ruido=RAIZ / "modelos" / "ruido.json")
    estado = {}
    correr.etapa_datos(estado, args)
    correr.etapa_embeddings(estado, args)
    correr.etapa_narrativas(estado, args)
    com = estado["datos"]["comentarios"]
    extra = estado["extra_narrativas"]
    bando_com, resumen = postura.calcular_bandos(com, estado["emb"], estado["emb_indice"], extra)
    # Señal de coordinación (la misma función que usa correr.py)
    grupo, _ = coordinacion.detectar(com)
    for k, votos in postura.bando_de_operaciones(com, bando_com, extra, grupo).items():
        resumen[k]["coordinadas"] = votos

    # Verdad (solo práctica): en cada narrativa, ¿qué bando tiene más comentarios "apoya"?
    ruta_verdad = carpeta_datos / "verdad" / "comentarios.csv.gz"
    if ruta_verdad.exists():
        verdad = pd.read_csv(ruta_verdad, dtype=str).set_index("comentario_id")
        ids = com["comentario_id"].to_numpy()
        narr = extra["narr_num"]
        for k, v in resumen.items():
            en = np.where((narr == k) & (bando_com >= 0))[0]
            real = verdad.reindex(ids[en])
            v["narrativa_real"] = Counter(real["narrativa_id"].dropna()).most_common(1)[0][0]
            apoyo = [float((real["postura"].to_numpy()[bando_com[en] == b] == "apoya").mean()) for b in (0, 1)]
            v["apoyo_real_por_bando"] = [round(a, 3) for a in apoyo]
            v["apoya_real"] = int(apoyo[1] > apoyo[0])       # el bit correcto
    return resumen


def _ruta_resumen(carpeta_cache, carpeta_datos):
    return carpeta_cache / f"polaridad_{carpeta_datos.name}.json"


def _preguntar_todas(resumen, modelo, prompt, repeticion, respuestas, ruta_resp):
    """Pregunta (o lee de la caché de experimentos) a `modelo` por cada narrativa, de las dos formas:
    con los bandos en orden y al revés. Devuelve {narrativa: [bando elegido en orden, bando elegido al revés]}
    (None donde no hubo respuesta)."""
    instrucciones = ia.PROMPTS[prompt]
    parametros = ia.parametros_gateway(modelo)

    def una(tarea):
        k, invertido = tarea
        mensaje = ia._mensaje(resumen[k], invertido)
        clave = cache_ia.huella(modelo, {**parametros, "voto": repeticion}, instrucciones, mensaje)
        if clave not in respuestas:
            entrada = ia._preguntar_gateway(k, mensaje, modelo, instrucciones)
            if entrada:
                respuestas[clave] = entrada
                ruta_resp.write_text(json.dumps(respuestas, ensure_ascii=False, indent=1), encoding="utf-8")
        e = respuestas.get(clave)
        if not e:
            return k, invertido, None, 0.0
        bando = "AB".index(e["respuesta"]["bando_que_afirma"])
        return k, invertido, (1 - bando if invertido else bando), e["usd"]

    votos, usd = {k: [None, None] for k in resumen}, 0.0
    # De a uno: el gateway gratis deja 5 pedidos por minuto por modelo (ia._esperar_turno hace la espera)
    for k, invertido, bando, costo in map(una, [(k, inv) for k in sorted(resumen) for inv in (False, True)]):
        votos[k][int(invertido)] = bando
        usd += costo
    return votos, usd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("accion", choices=["armar", "medir", "regla"])
    ap.add_argument("--datos", type=Path, default=RAIZ / "datos/practica")
    ap.add_argument("--cache", type=Path, default=RAIZ / "cache")
    ap.add_argument("--modelo", default=ia.MODELO_GATEWAY)
    ap.add_argument("--prompt", default=ia.PROMPT_ACTUAL, choices=sorted(ia.PROMPTS))
    ap.add_argument("--repeticiones", type=int, default=3)
    ap.add_argument("--modelos", default=",".join(ia.MODELOS_GATEWAY), help="regla: modelos que votan")
    ap.add_argument("--repeticion", type=int, default=0, help="regla: qué repetición de `medir` usar")
    args = ap.parse_args()
    ruta = _ruta_resumen(args.cache, args.datos)

    if args.accion == "armar":
        resumen = armar(args.datos, args.cache)
        ruta.write_text(json.dumps(resumen, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"Resumen de {len(resumen)} narrativas en {ruta}")
        for k, v in resumen.items():
            print(k, v.get("narrativa_real", "?"), v["tamano"], "coordinadas:", v.get("coordinadas"),
                  "apoya_real:", v.get("apoya_real", "?"))
        return

    resumen = json.loads(ruta.read_text(encoding="utf-8"))
    ia._elegir_proveedor()      # carga la clave del entorno o de .env (sin mostrarla)
    con_verdad = all("apoya_real" in v for v in resumen.values())

    if args.accion == "regla":
        # Regla final SIN llamar a nadie: junta las respuestas ya guardadas por "medir" de varios
        # modelos y aplica la misma función que el pipeline (ia.votos_por_modelo + ia._combinar).
        modelos = [m.strip() for m in args.modelos.split(",") if m.strip()]
        guardadas = {}
        for m in modelos:
            ruta_m = args.cache / f"polaridad_respuestas_{m.replace('/', '_')}.json"
            guardadas[m] = json.loads(ruta_m.read_text(encoding="utf-8")) if ruta_m.exists() else {}
        bien = 0
        for k, v in sorted(resumen.items()):
            respuestas = []
            for m in modelos:
                for invertido in (False, True):
                    clave = cache_ia.huella(m, {**ia.parametros_gateway(m), "voto": args.repeticion},
                                            ia.PROMPTS[args.prompt], ia._mensaje(v, invertido))
                    e = guardadas[m].get(clave)
                    if e:
                        bando = "AB".index(e["respuesta"]["bando_que_afirma"])
                        respuestas.append((m, 1 - bando if invertido else bando))
            votos, contradicciones = ia.votos_por_modelo(respuestas)
            final, motivo = ia._combinar(votos, v["tamano"], v.get("coordinadas"))
            mayor = 0 if v["tamano"][0] >= v["tamano"][1] else 1
            ok = ""
            if con_verdad:
                ok = "OK " if final == v["apoya_real"] else "MAL"
                bien += ok == "OK "
            print(f"{k} {v.get('narrativa_real', ''):3s} final={'AB'[final]} mayoría={'AB'[mayor]} {ok} "
                  f"votos={''.join('AB'[x] for x in votos) or '-'} se contradicen={contradicciones} | {motivo}")
        if con_verdad:
            print(f"Regla final con {'+'.join(modelos)} (prompt {args.prompt}, repetición {args.repeticion}): "
                  f"{bien} de {len(resumen)}")
        return

    # medir: N repeticiones del mismo modelo y prompt, cada una con las dos órdenes de los bandos
    ruta_resp = args.cache / f"polaridad_respuestas_{args.modelo.replace('/', '_')}.json"
    respuestas = json.loads(ruta_resp.read_text(encoding="utf-8")) if ruta_resp.exists() else {}
    total_usd = 0.0
    for r in range(args.repeticiones):
        votos, usd = _preguntar_todas(resumen, args.modelo, args.prompt, r, respuestas, ruta_resp)
        total_usd += usd
        # Tres formas de contar: solo orden normal, solo invertido, y "consistente" (vota solo si
        # las dos órdenes coinciden; si se contradice, se abstiene)
        normal = sum(v[0] == resumen[k]["apoya_real"] for k, v in votos.items()) if con_verdad else "?"
        invertido = sum(v[1] == resumen[k]["apoya_real"] for k, v in votos.items()) if con_verdad else "?"
        bien = [k for k, v in votos.items() if v[0] == v[1] is not None and con_verdad and v[0] == resumen[k]["apoya_real"]]
        mal = [k for k, v in votos.items() if v[0] == v[1] is not None and con_verdad and v[0] != resumen[k]["apoya_real"]]
        duda = [k for k, v in votos.items() if v[0] != v[1]]
        nombre = lambda ks: ",".join(f"{k}({resumen[k].get('narrativa_real', '')})" for k in sorted(ks)) or "-"
        print(f"{args.modelo} prompt={args.prompt} rep={r}: orden normal {normal}/13, invertido {invertido}/13 | "
              f"consistente: {len(bien)} bien, {len(mal)} mal [{nombre(mal)}], {len(duda)} se contradice "
              f"[{nombre(duda)}] | votos " + " ".join(k + ":" + "".join("-" if b is None else "AB"[b] for b in v)
                                                     for k, v in sorted(votos.items())), flush=True)
    print(f"{args.modelo} prompt={args.prompt}: US$ de lista {total_usd:.4f}", flush=True)


if __name__ == "__main__":
    main()
