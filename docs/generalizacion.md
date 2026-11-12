# ¿Generaliza? Simulación del set final con el set de práctica

El set final tiene **otros temas, otras narrativas y otras campañas**. Para estimar cómo nos va ahí,
simulamos esa situación con el set de práctica y medimos todo con el script oficial del jurado.
Todo se corre con un comando (`uv run python evaluacion/generalizacion.py`, unos 15 minutos) y los
números quedan en [`experimentos/generalizacion.csv`](../experimentos/generalizacion.csv).
Todas las corridas son sin IA y con el mismo código que generó la entrega.

## Resultados

| Experimento | Narrativas /30 | B-cubed | Ruido F1 | Postura F1 | Origen | Coordinación /25 | Total /55 |
|---|---|---|---|---|---|---|---|
| **Referencia**: práctica completa | 29,00 | 0,981 | 0,952 | 0,916 | 1,00 | 22,98 | 51,98 |
| Tema afuera: **colectivo** | 28,13 | 0,981 | 0,961 | 0,766 | 1,00 | (ver abajo) | — |
| Tema afuera: **hospital** | 29,24 | 0,973 | 0,942 | 0,986 | 1,00 | (ver abajo) | — |
| Tema afuera: **litio** | 28,72 | 0,959 | 0,872 | 0,984 | 1,00 | 23,80 | 52,52 |
| Solo los **primeros 10 días** | 29,14 | 0,983 | 0,960 | 0,930 | 1,00 | 22,77 | 51,91 |
| Semilla 1 | 29,00 | 0,981 | 0,953 | 0,916 | 1,00 | 22,90 | 51,90 |
| Semilla 2 | 28,99 | 0,981 | 0,952 | 0,916 | 1,00 | 23,20 | 52,19 |
| Semilla 3 | 28,62 | 0,978 | 0,949 | 0,862 | 1,00 | 22,55 | 51,17 |
| **Datos sucios** | 29,16 | 0,970 | 0,938 | 0,982 | 1,00 | 22,98 | 52,14 |

### Cómo se armó cada experimento

- **Tema afuera:** para cada tema (litio, colectivo, hospital) se entrena el clasificador de ruido
  **solo con los otros dos** y se corre el pipeline **solo** sobre las publicaciones del tema que quedó
  afuera. Es lo más parecido al set final: un tema que el pipeline nunca vio.
- **Primeros 10 días:** el set final dura 10 días; práctica, 14.
- **Semillas:** la misma corrida con semillas 1, 2 y 3 en vez de 42 (UMAP, Leiden y muestreos).
- **Datos sucios:** columnas al revés, una columna extra, 5% de filas duplicadas, 1% de fechas rotas,
  1% de textos vacíos y 10% de cuentas sin metadata.

### Coordinación cuando se deja un tema afuera

El F1 de cuentas no sirve tal cual en los subconjuntos por tema, porque las cuentas de campaña también
comentan "de cobertura" en temas que no son el suyo (1 a 8 comentarios sueltos). Lo que importa es
si encontramos la campaña que **opera** en ese tema y si marcamos a alguien que no lo es:

| Tema afuera | Campaña que opera ahí | Encontradas | Cuentas genuinas marcadas |
|---|---|---|---|
| colectivo | ninguna | — | **0** de 13.558 |
| hospital | C3 (25 cuentas) | **23 de 25** | **0** de 13.085 |
| litio | C1 (60) + C2 (30) | **84 de 90** (58 + 26) | 1 de 14.140 |

## Qué dicen los números

1. **Las narrativas generalizan.** Con un tema nunca visto, el puntaje de narrativas queda entre
   28,13 y 29,24 (promedio 28,70 contra 29,00): una brecha de **−0,3 puntos (−1%)**. El agrupamiento
   (B-cubed 0,96 a 0,98) y el origen de las emergentes (3 de 3 siempre) no dependen del tema.
2. **Lo que más se resiente es el ruido** cuando el clasificador no vio el tema (litio: 0,872). Es la
   única pieza entrenada con la verdad, y aun así se mantiene por encima de 0,87.
3. **La coordinación no inventa operaciones:** en el tema sin campañas no marca a nadie, y en los otros
   la precisión es de 0,99 a 1,00.
4. **La postura sin IA es la parte más frágil.** La regla "el bando más grande apoya" falla en la
   narrativa B3 (por eso colectivo da 0,766). Con datos sucios o con 10 días el tamaño de los bandos
   cambia y B3 a veces sale bien (0,93 a 0,98). Es el lugar donde la IA opcional suma: 1 llamada por
   narrativa decide qué bando afirma.
5. **Semillas:** el total varía entre 51,17 y 52,19 (±1%). La entrega usa siempre la semilla 42 y
   da exactamente el mismo resultado en cada corrida.
6. **Datos sucios:** el pipeline no se cae y el puntaje no baja (52,14).

## Hiperparámetros: ¿salen de los datos o de práctica?

| Parámetro | Valor | ¿Relativo a los datos? |
|---|---|---|
| Tamaño mínimo de micro-grupo | 0,1% de los textos (mínimo 15) | Sí, escala con el set |
| Cantidad de narrativas | La decide Leiden con los votos de los hilos | Sí, no se fija |
| Umbral de ruido | Clasificador con rasgos sin tema (largo, parecido, margen, pregunta) | Entrenado en práctica, validado por tema |
| Rescate de textos sueltos | Percentil 5 de parecido de cada micro-grupo | Sí, por grupo |
| Fecha de origen | ≥5 comentarios del núcleo en 6 h | Fijo, pero no depende del tema |
| Coordinación: repetición | ≥3 publicaciones y ≥30% de la actividad de la cuenta | Fijo (es la definición de "evidencia repetida") |
| Coordinación: franja | Se ignoran momentos con >30 cuentas en 15 min (virales) | Fijo |
| Red de respuestas | ≥1,6 conexiones por cuenta (un árbol tiene ~1) | Relativo a la estructura de árbol |
| Hashtag propio | ≥2/3 de los usos de una operación | Relativo a la operación |
