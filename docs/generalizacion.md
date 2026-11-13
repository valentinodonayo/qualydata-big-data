# ¿Generaliza? Simulación del set final con el set de práctica

El set final tiene **otros temas, otras narrativas y otras campañas**. Para estimar cómo nos va ahí,
simulamos esa situación con el set de práctica y medimos todo con el script oficial del jurado.
Todo se corre con un comando (`uv run python evaluacion/generalizacion.py`, tarda varios minutos) y
los números quedan en [`experimentos/generalizacion.csv`](../experimentos/generalizacion.csv).

Esta tabla se regeneró el 13/11 con todo lo de hoy integrado (ruido por vecinos en hilos y, en coordinación, capa G,
filtro de par único y capa H de tanda de creación) **menos la IA**: todas las corridas usan `--sin-ia`, así que la polaridad de cada
narrativa la decide "el bando más grande".

## Resultados (sin IA)

| Experimento | Narrativas /30 | B-cubed | Ruido F1 | Postura F1 | Origen | Coordinación /25 | Total /55 |
|---|---|---|---|---|---|---|---|
| **Referencia**: práctica completa | 29,23 | 1,000 | 1,000 | 0,872 | 1,00 | 24,64 | 53,87 |
| Tema afuera: **colectivo** | 25,79 | 1,000 | 1,000 | 0,298 | 1,00 | (ver abajo) | — |
| Tema afuera: **hospital** | 30,00 | 1,000 | 1,000 | 0,999 | 1,00 | (ver abajo) | — |
| Tema afuera: **litio** | 29,83 | 0,989 | 1,000 | 0,999 | 1,00 | 24,72 | 54,55 |
| Solo los **primeros 10 días** | 29,57 | 0,996 | 1,000 | 0,938 | 1,00 | 24,86 | 54,43 |
| Semilla 1 | 28,62 | 1,000 | 1,000 | 0,770 | 1,00 | 24,42 | 53,04 |
| Semilla 2 | 29,23 | 1,000 | 1,000 | 0,872 | 1,00 | 24,72 | 53,95 |
| Semilla 3 | 29,22 | 0,999 | 1,000 | 0,872 | 1,00 | 24,34 | 53,56 |
| **Datos sucios** | 28,36 | 0,989 | 0,985 | 0,766 | 1,00 | 24,64 | 53,00 |

Con IA, la práctica completa da 54,63 (narrativas 29,99, postura 0,998). El hold-out no se corrió
con IA: cada experimento arma narrativas distintas y habría que volver a preguntarle a los modelos.

### Cómo se armó cada experimento

- **Tema afuera:** para cada tema (litio, colectivo, hospital) se corre el pipeline **solo** sobre las
  publicaciones de ese tema. El clasificador de ruido se entrena **solo con los otros dos**, aunque
  hoy casi no pesa: el ruido lo decide la regla de vecinos en hilos, que no usa etiquetas, y el
  clasificador queda de respaldo. Es lo más parecido al set final: un tema que el pipeline nunca vio.
- **Primeros 10 días:** el set final dura 10 días; práctica, 14.
- **Semillas:** la misma corrida con semillas 1, 2 y 3 en vez de 42 (UMAP, Leiden y muestreos).
- **Datos sucios:** columnas al revés, una columna extra, 5% de filas duplicadas, 1% de fechas rotas,
  1% de textos vacíos y 10% de cuentas sin metadata.

### Coordinación cuando se deja un tema afuera

El F1 de cuentas no sirve tal cual en los subconjuntos por tema, porque las cuentas de campaña también
comentan "de cobertura" en temas que no son el suyo (unos pocos comentarios sueltos), y el script las
cuenta como cuentas a encontrar. Lo que importa es si encontramos la campaña que **opera** en ese tema
y si marcamos a alguien que no lo es:

| Tema afuera | Campaña que opera ahí | Cuentas marcadas | De campaña | Genuinas marcadas |
|---|---|---|---|---|
| colectivo | ninguna | 0 | — | **0** |
| hospital | C3 (25 cuentas) | 36 | **36** (las de C3 y cuentas de las otras campañas que comentan ahí de cobertura) | **0** |
| litio | C1 (60) + C2 (30) | 90 | **89** (de 90) | 1 |

## Qué dicen los números

1. **Agrupamiento y ruido mejoran o empatan en los 9 experimentos** respecto de la versión de ayer.
   B-cubed queda entre 0,989 y 1,000. El ruido da F1 1,000 en 8 de 9; el único que baja es datos
   sucios (0,985). La regla nueva de ruido no usa etiquetas ni depende del tema, y se nota: con un
   tema nunca visto ya no hay caída (antes litio daba 0,872).
2. **El origen** de las emergentes da 3 de 3 en todos los experimentos.
3. **La postura sin IA es la parte frágil, y en 3 experimentos empeoró fuerte:** tema colectivo
   (0,298), semilla 1 (0,770) y datos sucios (0,766). En la referencia y en la semilla 2 también bajó
   un poco (de 0,916 a 0,872). No es que los bandos estén mal armados: en esas narrativas el apoyo
   real es de ~50%, y "el bando más grande apoya" es una moneda al aire. Un cambio chico en qué
   comentarios entran a la narrativa alcanza para dar vuelta cuál bando queda más grande, y con eso
   toda la narrativa queda con la postura invertida.
4. **Con la polaridad correcta, narrativas da entre 29,7 y 30,0 en los 9 experimentos.** Es una
   cuenta, no una corrida: la postura vale 6 de los 30 puntos (20%), así que con postura ≈ 1,0 cada
   fila sube 6 × (1 − postura F1). Elegir bien la polaridad es justo lo que hace la IA: en la
   práctica completa acierta 13 de 13 narrativas y lleva la postura de 0,872 a 0,998.
5. **La coordinación no inventa operaciones:** en el tema sin campañas no marca a nadie, y en hospital
   y litio la precisión es de 0,99 a 1,00. Con otras semillas la precisión baja algo (0,966 con la
   semilla 1 y 0,942 con la semilla 3; 1,000 con la 2).
6. **Semillas:** el total sin IA varía entre 53,04 y 53,95. La entrega usa siempre la semilla 42.
7. **Datos sucios:** el pipeline no se cae. El total (53,00) baja contra la referencia (53,87) por la
   postura sin IA; agrupamiento, ruido y coordinación casi no se mueven.

## Hiperparámetros: ¿salen de los datos o de práctica?

| Parámetro | Valor | ¿Relativo a los datos? |
|---|---|---|
| Tamaño mínimo de micro-grupo | 0,1% de los textos (mínimo 15) | Sí, escala con el set |
| Cantidad de narrativas | La decide Leiden con los votos de los hilos | Sí, no se fija |
| Ruido | Un comentario suelto es ruido si sus ~50 vecinos más parecidos participan de hilos menos del 10% de la tasa normal del set | Sí: el umbral es relativo a la tasa de hilos del propio set. No usa etiquetas |
| Ruido, respaldo | Clasificador entrenado en práctica (largo, parecido, margen, pregunta) | Solo se usa si menos del 2% de los comentarios está en hilos |
| Rescate de textos sueltos | Narrativa de sus copias en hilos o, si no hay, del micro-grupo más parecido | Sí, por texto |
| Fecha de origen | ≥5 comentarios del núcleo en 6 h | Fijo, pero no depende del tema |
| Coordinación: repetición | ≥3 publicaciones y ≥30% de la actividad de la cuenta | Fijo (es la definición de "evidencia repetida") |
| Coordinación: franja | Se ignoran momentos con >30 cuentas en 15 min (virales) | Fijo |
| Red de respuestas | ≥1,6 conexiones por cuenta (un árbol tiene ~1) | Relativo a la estructura de árbol |
| Hashtag propio | ≥2/3 de los usos de una operación | Relativo a la operación |
| Llegadas raras (capa G) | Puntaje ≥ 1,4, densidad ≥ 0,03 por publicación de la cuenta, ≥3 publicaciones en común | Fijo, elegido en práctica. El peso 1/(presentes − 1) sí es relativo a cuánta gente hay |
| Polaridad con IA | IA si ≥2 modelos votan y coinciden; si no, puntos (operación con ≥80% en un bando, bando más grande con ≥54%) | Fijo, elegido en práctica |
