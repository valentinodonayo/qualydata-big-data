# Mejoras de la sesión del 12/11 (21:15 a 22:45): qué se aceptó y qué se descartó

Regla: un cambio se acepta solo si mejora (o no empeora) **en las dos** validaciones: (a) el set de
práctica completo con el script oficial y (b) el hold-out (`evaluacion/generalizacion.py`: tema dejado
afuera, primeros 10 días, otras semillas, datos sucios). Todo sin IA, porque en esta sesión no hay
clave de API. Diagnóstico de partida: [`diagnostico.md`](diagnostico.md).

## Cambios medidos

| Cambio | Práctica antes → después | Hold-out | Decisión |
|---|---|---|---|
| Coordinación, capa H "tanda de creación": si una operación detectada se creó en pocos días (percentiles 5-95 en ≤ 10 días) con la misma forma de usuario (≥ 80%, ej. palabra+4 dígitos), sumar las cuentas no marcadas creadas en esa ventana (±1 día) con esa forma. Sin fechas ni nombres escritos a mano; ignora cuentas sin metadata | 53,35 → **53,64** (cuentas F1 0,978 → 0,987, precisión igual 0,991, cobertura 0,983) | Mejora o empata en los 8: hospital afuera +2,79, litio +0,36, primeros 10 días +0,87, semillas +0,29/+0,29/+0,28, datos sucios +0,29 (10% de cuentas sin metadata), colectivo igual. Precisión nunca baja | **Aceptado**. La parte "quitar miembros fuera de la tanda" no se implementó: no se disparaba en ningún set |
| Coordinación A-C: soltar a la cuenta que entra por UN solo par fuerte cuando ese par es < 10% de su propia actividad (cuenta muy activa que coincide por volumen) | 53,28 → **53,35** (precisión 0,982 → 0,991, sin perder reales; igual con 5%, 10% y 20%) | Mejora o empata en los 8 (semillas +0,07/+0,08/+0,07, datos sucios +0,07, el resto igual). Precisión nunca baja | **Aceptado** |
| Capa G, segunda vuelta: mínimo 2 publicaciones en común (antes 3) y hasta 3 pasadas (una cuenta recién sumada puede delatar a otra) | 52,99 → **53,28** (cuentas F1 0,965 → 0,974, precisión igual 0,982; 113 marcadas) | Mejora o empata en los 8: hospital +0,47, litio y colectivo igual, primeros 10 días +0,76, semillas +0,29/+0,29/+0,58, datos sucios +0,29. Precisión nunca baja | **Aceptado** |
| Coordinación, capa G "llegadas raras": sumar a una operación la cuenta que llega a sus publicaciones en momentos tranquilos (cada coincidencia en 15 min pesa 1/(presentes−1); puntaje ≥ 1,4, densidad ≥ 0,03 por publicación, ≥ 3 publicaciones en común) | 51,98 → **52,99** (cuentas F1 0,932 → 0,965, precisión 0,981 → 0,982; grupos 0,891 → 0,948). Estable: rareza 1,2–1,4 y densidad 0,02–0,05 dan lo mismo; con 1,6, 52,56 | Mejora o empata en los 8: tema afuera hospital +0,47, litio +0,56, colectivo igual, primeros 10 días +0,46, semillas +0,86/+0,87/+0,86, datos sucios +1,01. Precisión nunca baja | **Aceptado**. +0,4 s por corrida. Set final: +3 cuentas en op02 |
| Coordinación: el par de cuentas tiene que coincidir en ≥30% de la actividad de **las dos** cuentas (antes, de la menos activa) | 51,98 → **52,06** (precisión de cuentas 0,981 → 0,990; recall igual) | Mejora semillas (+0,07) y datos sucios (+0,08), pero **empeora** tema afuera litio (52,52 → 51,97: pierde 3 cuentas reales de campaña) y primeros 10 días (51,91 → 51,76) | **Revertido** (regla madre). Resultados en `experimentos/generalizacion_pares_descartado.csv` |
| Postura sin IA: "apoya" = bando con menos negaciones | Acierta 8 de 13 narrativas (hoy: 12 de 13) | — | **Descartado** sin implementar |
| Postura sin IA: "apoya" = bando al que más le responden | 9 de 13 | — | **Descartado** |
| Postura sin IA: "apoya" = bando con más raíces | 12 de 13 (igual que hoy) | — | Descartado: no mejora |
| Coordinación: sumar cuentas que llegan cerca de una operación más de lo esperable por azar (modelo nulo de Poisson, ventanas de 15, 5 y 2 min) | Ninguna cuenta supera lo esperado, ni siquiera las 9 de C2 que se escapan | — | **Descartado**: la señal no existe en estos datos |
| Coordinación, capa F "paráfrasis": sumar a una operación la cuenta que dice lo mismo que un miembro (embeddings, similitud ≥ 0,93) en ≥ 5 publicaciones | Medición previa: las 9 cuentas de C2 que se escapan **no** parafrasean a nadie de su campaña, y 630 genuinas tienen tanta paráfrasis como las de campaña. Solo usando los 10 vecinos más parecidos de cada texto recupera 2 cuentas de C3 sin falsos positivos; implementada sin ese filtro (para no comparar todo contra todo) suma 3.927 cuentas y la práctica cae a 29,38 | No hizo falta | **Revertido**: la versión que funciona agrega ~35 s por corrida (+30%) para ganar ~0,15 puntos en práctica, y depende de un parámetro frágil (cuántos vecinos). Práctica después de revertir: 51,98 |
| Tiempo: UMAP con 100 iteraciones en vez de 200 | Aislado, UMAP 60 s → 38 s; pero en el pipeline completo la práctica **baja de 51,98 a 51,70** (otra narrativa queda con la postura invertida) y la etapa no bajó de tiempo | No hizo falta | **Revertido** |

## Cambios que no tocan el puntaje (infraestructura)

| Cambio | Por qué | Verificación |
|---|---|---|
| Caché versionada de la IA (`modelos/cache_ia.json`, `pipeline/cache_ia.py`) | El jurado re-corre sin nuestra clave. Con la caché, la entrega con IA se reproduce idéntica, sin clave ni internet. | Tests sin llamar al modelo: acierto exacto, respaldo sin clave, respaldo "parecido" con bandos invertidos |
| IA gratis: Google Gemini 3.8 Flash como primer proveedor (`GEMINI_API_KEY`), Claude Haiku queda como alternativa | La cuenta de Anthropic no tenía saldo (la API respondía "credit balance is too low"). Gemini tiene capa gratis sin tarjeta, y con la caché el jurado no necesita ninguna clave | 4 tests nuevos sin internet: elige Gemini primero, prueba otra forma del pedido si Google responde 400, guarda en la caché y reproduce sin clave, y usa el respaldo si la API falla. Práctica sin clave que funcione: 51,98 (igual) |
| Ejemplos para la IA = las 12 raíces **más centrales** de cada bando (antes, 12 al azar) | Estables entre máquinas (GPU vs CPU cambian decimales). Si igual cambia alguno, la caché reconoce cada bando por ejemplos en común | Sin IA el resultado es idéntico (la elección de ejemplos solo afecta al pedido a la IA) |

## Por qué no se agregó una cascada NLI + LLM para la postura

El diagnóstico muestra que la postura **por comentario** ya es casi perfecta: dentro de cada
narrativa, los dos bandos (armados con las respuestas que se contradicen) aciertan el 98,7% al
100%. Lo único que falla es **un bit por narrativa**: cuál de los dos bandos afirma. Ese bit
ya lo decide una sola llamada a la IA por narrativa (~12 por corrida). Un clasificador NLI
comentario por comentario agregaría costo y tiempo para resolver algo que ya está resuelto.
Sin clave, ninguna señal general (negación, dirección de respuestas, raíces) le gana a la regla
actual. Y es frágil: los dos bandos tienen casi el mismo tamaño, así que un cambio mínimo en los
grupos (como el de UMAP con menos iteraciones) invierte otra narrativa. Es el riesgo principal
que queda para el set final (ver "Límites conocidos" en el README).

## Costo y tiempo

| Medida | Antes | Después |
|---|---|---|
| Segundos por cada 1.000 comentarios (set final, cache vacía, GPU de la Mac) | 0,74 s | 0,74 s (sin cambios de cómputo aceptados) |
| USD de IA por cada 1.000 comentarios | 0 (sin clave) | 0 (sin clave). Con clave: ~12 llamadas a Haiku 5.5 ≈ US$ 0,002 por corrida (≈ US$ 0,000013 por cada 1.000); con la caché, US$ 0 en cada re-corrida |
| Cuello de botella medido | UMAP + HDBSCAN: 61 s de 111 (55%); embeddings 29 s; cargar el modelo 6 s | Igual. UMAP corre en un solo hilo porque fijar la semilla lo exige: paralelizarlo rompería la reproducibilidad |
