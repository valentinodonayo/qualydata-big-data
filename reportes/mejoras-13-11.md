# Sesión del 13/11 (mañana): herencia de postura y por qué las 2 cuentas de C2 son irrecuperables

Regla igual que ayer: un cambio se acepta solo si mejora (o no empeora) en el set de práctica con el
script oficial **y** en los sets de validación, todo sin IA. Las corridas quedan en
`experimentos/runs.csv` (prefijo `devin-`).

## Aceptado: dos excepciones medidas a la herencia de postura por hilo

La regla era "respuesta corta = misma postura que el padre; larga = opuesta". Midiendo contra la
verdad, 150 comentarios heredaban mal y caían en dos clases:

- **Respuestas cortas que contradicen** (45): llevan contenido propio y el mismo texto aparece como
  raíz. Si un texto corto aparece ≥ 2 veces como raíz con ≥ 70% de una postura, la respuesta toma
  esa postura y no la del padre.
- **Respuestas largas que acuerdan** (35, casi todas plantillas "Eso, …", "Tal cual, …"): si la
  respuesta abre con una fórmula de acuerdo (`tal cual`, `eso`, `exacto`, `de acuerdo`, `obvio`,
  `claro que sí`, …) sin negación pegada, acompaña al padre aunque sea larga.

| Medición | Resultado |
|---|---|
| Simulación con la verdad | 145 de 150 errores corregidos, 0 regresiones |
| En producción (diff entre entregas) | 152 comentarios cambian: **141 mejoran, 11 empeoran** (8 son artefacto del modo sin IA: la polaridad invertida de B3 arrastra la regla texto-raíz; con IA esas 8 vuelven) |
| Práctica oficial, sin IA | 53,95 → **53,96** (postura 0,872 → 0,873; el resto del efecto queda tapado por la polaridad de B3, que la IA resuelve: con IA la postura ya es 0,998) |
| Inventado `gemelo` | 42,88 → 42,88 |
| Inventado `dif_a` (15% de respuestas largas **acuerdan**: la variante que ejercita esta regla) | 38,24 → **38,32** (postura 0,976 → 0,988) |
| Inventados `dif_c`, `dif_g` | 38,60 y 41,26, sin cambio |

Detalle de orden: la fórmula de acuerdo se evalúa **antes** que la regla texto-raíz, porque una
respuesta "Tal cual, X" acompaña al padre aunque ese mismo texto tenga mayoría propia como raíz.

## Descartado: todo lo que se probó para las 2 cuentas de C2 que se escapan

Las 2 (`u003682`, `u009692`) son de la campaña "de cobertura": cuentas viejas, textos variados,
cada una intercambia solo 2 respuestas con la operación y comenta en publicaciones populares en
horarios normales. Se probaron 7 enfoques, incluidos 2 tomados de la literatura 2025:

| Enfoque | Resultado | Por qué no |
|---|---|---|
| Solape de publicaciones con la operación | Las fugitivas comparten 60–70% de sus pubs con socios | Más de 5.000 genuinas cumplen lo mismo (la campaña va a las publicaciones populares) |
| Ventanas temporales (2 a 24 h) | Cada fugitiva comparte ventanas con 25–29 de los 29 socios | 6.978 genuinas también |
| Brecha de llegada a pubs compartidas | — | No separa: la sincronización real no está en escala de horas |
| Peso por dominancia de la publicación | La peor fugitiva queda en el puesto 640 del ranking | ~600 genuinas la superan |
| Relajar la capa E (≥2 respuestas con ≥2 miembros) | Recupera `u009692` | Suma 4 genuinas: F1 de cuentas 0,983 < 0,991 actual |
| **NCD por compresión** (Loru et al. 2025, DOI 10.1145/3778356): distancia de compresión entre la secuencia de publicaciones de cada cuenta y la traza agregada de la operación | Fugitivas en puestos 132 y 144 de ~1.600 (percentil 8,7) | La señal existe pero ~130 genuinas son "más campaña" que ellas |
| **Ráfagas compartidas + correlación temporal** (arXiv:2504.02757): fracción de actividad dentro de los picos de la operación y correlación de su serie horaria con la de la operación | `u009692`: p99,7 en correlación; ambas > p95 en fracción en ráfagas | Combinadas sobre las candidatas con algún vínculo débil, el top-12 del ranking es 100% genuinas: las ráfagas de la operación coinciden con las de los hilos populares donde también está la gente común |

**Conclusión:** en 7 enfoques las 2 cuentas son estadísticamente indistinguibles de genuinas
hiperactivas. El diseño de "cobertura" del desafío funciona como se esperaba. Recuperarlas cuesta
0,28 puntos de cobertura y baja la precisión; no rinde.

## Cómo encaja el diseño con la literatura reciente

| Trabajo | Idea | Dónde está en el pipeline |
|---|---|---|
| arXiv:2512.19677 (dic-2025) | Red multiplex + colaboración temporal con decaimiento exponencial | Descomposición por capas + capa G (cada coincidencia pesa 1/presentes: la coincidencia "rara" vale más) |
| arXiv:2506.13912 | Paseos density-aware + embeddings estructurales para campañas | Capa D: ciclos en la red de respuestas contra árboles (la versión práctica de la idea estructural, sin entrenar embeddings) |
| arXiv:2410.19272 | Detección por ataques de respuesta coordinada | Capa D es la versión no supervisada de la misma señal |
| Loru et al. 2025 (NCD) y arXiv:2603.03128 (BLOC/cambio conductual) | Similitud de trazas conductuales por compresión | Probado en esta sesión: no separa a las de cobertura (tabla de arriba) |

La única dirección de la literatura no cubierta por las 7 capas es la **influencia causal** entre
series temporales (CCM, arXiv:2407.11690; AMDN-HAGE, arXiv:2008.11308; VigDet, arXiv:2110.15454):
modelar si la actividad de una cuenta *dispara* la de la operación, en vez de medir co-presencia.
No se implementó: son modelos pesados (redes + inferencia variacional) sin ventaja demostrable en
un set de 14 días, y la precisión actual es 0,99–1,00.
