# Costo y escala: de 150.000 comentarios a millones por día

Todos los números de este documento salen de mediciones propias en una MacBook (Apple M5,
10 núcleos, 16 GB de RAM), salvo donde dice **supuesto**. Los tiempos y el costo de IA del set final
salen de [`entrega/costo.json`](../entrega/costo.json); el resto, de los scripts de `evaluacion/`.

## 1. Lo que cuesta hoy (set final, 150.000 comentarios)

Corrida desde cero (sin cache), con los embeddings en la GPU integrada de la Mac:

| Etapa | Segundos | Qué hace |
|---|---|---|
| Datos | 0,5 | Leer CSV.gz, limpiar, guardar DuckDB + Parquet |
| Embeddings | 39,0 | Solo los textos distintos (alrededor de un tercio de los comentarios) |
| Narrativas | 68,0 | UMAP + HDBSCAN + votos de hilos + Leiden + ruido por vecinos |
| Coordinación | 4,5 | Pares de cuentas + red de respuestas + hashtag propio + llegadas raras |
| Postura | 0,8 | 2 bandos por narrativa + regla de polaridad (lee la caché de IA) |
| **Total** | **113** | Solo con CPU tarda más: los embeddings son la etapa que más se alarga |

- **IA:** con el gateway son **6 llamadas por narrativa** (3 modelos × 2 órdenes de los bandos). En
  el set final: 72 respuestas, 117.915 tokens y **US$ 0,187 a precio de lista**, o sea
  **US$ 0,0012 por cada 1.000 comentarios** y unos US$ 0,016 por narrativa. El costo **no crece con
  el volumen**: depende de cuántas narrativas hay, no de cuántos comentarios. Cada re-corrida cuesta
  **US$ 0**, porque lee las respuestas de la caché versionada (`modelos/cache_ia.json`). El precio
  por token de cada modelo está en `pipeline/ia.py`; la cuenta usada está en la capa gratis, así que
  `usd_ia` es lo que costaría pagarlo.
- **Cómputo:** ≈0,75 ms por comentario (113 s para 150.000). Con una VM de 8 a 10 núcleos
  a US$ 0,30 por hora (**supuesto**, precio típico on-demand), son ≈1 s por cada 1.000 comentarios,
  o sea **≈US$ 0,0001 por cada 1.000 comentarios**.
- **Almacenamiento:** el set de práctica (200.000 comentarios) pesa 4,6 MB en CSV.gz, 14 MB como
  base DuckDB (con índices y tipos) y 4,7 MB en Parquet. Los embeddings son 384 números de 4 bytes:
  1,5 KB por texto distinto (≈75 MB para el set final).

### Por qué DuckDB (medido)

Leer los 200.000 comentarios, contar por publicación y por día, y guardar en Parquet
(`evaluacion/medir_almacenamiento.py`, segunda corrida de cada uno):

| Motor | Segundos | Memoria máxima | Parquet |
|---|---|---|---|
| DuckDB | 0,36 | 241 MB | 7,5 MB (snappy) |
| Polars | 0,35 | 267 MB | 3,3 MB (zstd) |
| pandas | 0,96 | 212 MB | 7,6 MB |

DuckDB y Polars empatan en velocidad. Elegimos DuckDB porque además es **una base en un archivo,
sin servidor** (el jurado la re-crea con un comando), habla **SQL**, **aparta las filas rotas**
(`store_rejects`) en vez de cortarse, y procesa datos **más grandes que la memoria**. De Polars
copiamos la idea de comprimir el Parquet con zstd.

## 2. Qué se rompe al crecer

| Pieza | Cómo crece | Problema a millones por día |
|---|---|---|
| Embeddings | Lineal en textos **distintos** (los repetidos se reutilizan) | Ninguno: se paraleliza por lotes |
| UMAP + HDBSCAN | Más que lineal y en memoria | **El cuello de botella**: no se puede re-agrupar todo cada vez |
| Votos de hilos + Leiden | Lineal en respuestas | Ninguno (grafo de cientos de nodos) |
| Ruido por vecinos | Cada texto suelto se compara contra todos los textos distintos (producto de matrices, por tandas) | Crece cuadrático: a esa escala los vecinos se buscan en el índice ANN |
| Coordinación (pares) | Cuadrático **dentro** de cada franja de 15 min | Acotado: las franjas con >30 cuentas se ignoran |
| Coordinación (red de respuestas) | Lineal en respuestas | Ninguno |
| Coordinación (llegadas raras) | Lineal en comentarios (se agrupa por publicación y franja) | Ninguno |
| IA | 6 llamadas por narrativa, no por comentario | Ninguno: no depende del volumen |

## 3. Diseño para millones de comentarios por día

```
 fuentes ──► Redpanda/Kafka ──► workers de embeddings (lotes, ONNX/GPU) ──► ClickHouse
                                         │                                    ▲
                                         ▼                                    │
                              índice ANN de centros ──► asignación en vivo ───┘
                              (HNSW, una entrada por        (narrativa + postura
                               micro-grupo)                  por hilo)
                                         ▲
             cada 1 h: HDBSCAN solo sobre los comentarios "sin narrativa" de la ventana
             ──► narrativas NUEVAS (novedad) ──► se agregan al índice; Leiden con votos de hilos
```

1. **Ingesta:** Redpanda o Kafka (particionado por publicación, así un hilo queda en una partición).
2. **Almacenamiento:** ClickHouse, el mismo modelo columnar que DuckDB pero distribuido. Tablas
   `comentarios` (particionada por día, ordenada por publicación y fecha) y `asignaciones`. DuckDB
   sigue sirviendo para análisis locales sobre el Parquet exportado.
3. **Asignación en vivo:** cada comentario nuevo se compara contra los centros de los micro-grupos
   con un índice ANN (HNSW). **Medido** (`evaluacion/medir_streaming.py`): con centros armados solo
   con los días 1 a 7 de práctica, asignar las raíces de los días 8 a 14 al centro más cercano
   acierta el **92,8%** en las narrativas que ya existían, a **0,27 ms por cada 1.000 comentarios**.
4. **Novedad:** lo que la asignación en vivo no ve son las narrativas que nacen después (en esa
   prueba, el 10,8% de las raíces eran de una narrativa nueva). Por eso, cada hora se corre HDBSCAN
   **solo** sobre los comentarios que quedaron lejos de todo centro. Esa es la parte cara, pero
   sobre una fracción chica de los datos. Si aparece un grupo nuevo con votos de hilos, es una
   narrativa nueva y su fecha de origen sale del mismo estimador que hoy.
5. **Coordinación en streaming:** contadores por ventana de 15 min y por publicación (ClickHouse
   `AggregatingMergeTree` o Flink). Los pares que se repiten en 3 o más publicaciones distintas
   pasan al grafo. Para las copias exactas o casi exactas se usa MinHash + LSH en vez de comparar
   todos contra todos.
6. **Ruido en vivo:** la regla de hoy (¿los vecinos más parecidos participan de hilos?) se
   responde con el mismo índice ANN, guardando por cada texto cuántas copias tiene y cuántas están
   en hilos. Es un diseño, no está medido.
7. **IA:** siguen siendo 6 llamadas por narrativa **nueva**, no por comentario. El gasto de IA no
   depende del volumen sino de cuántas narrativas aparecen, y las respuestas quedan en la caché.

### Cuánto costaría (con nuestras mediciones)

Para **1 millón de comentarios por día**, con ~33% de textos distintos (lo medido en los dos sets):

| Concepto | Cálculo | Por día |
|---|---|---|
| Embeddings en CPU | 330.000 textos × 2,6 ms (medido el 12/11) | ≈14 min de una VM |
| Asignación ANN | 1.000.000 × 0,27 µs | < 1 s |
| Re-agrupado horario | ~10% sin narrativa = 100.000/día, en 24 lotes de ~4.000 | ≈5 min de una VM (**estimado**: hoy la etapa de narrativas completa tarda 68 s para el set final; se toma un margen amplio) |
| Coordinación | lineal, 4,5 s por cada 150.000 | ≈30 s |
| **Total cómputo** | ≈20 min de una VM de 8 a 10 núcleos a US$ 0,30/h (**supuesto**) | **≈US$ 0,10** |
| IA | ~10 narrativas nuevas por día (**supuesto**) × US$ 0,016 a precio de lista | ≈US$ 0,16 |

El cómputo da **≈US$ 0,0001 por cada 1.000 comentarios**: una sola VM chica alcanza para varios
millones por día. La IA suma unos US$ 0,16 por día si aparecen 10 narrativas nuevas, sin importar
cuántos comentarios lleguen: a ese volumen, menos de US$ 0,0002 por cada 1.000 comentarios. Lo que más crece es el almacenamiento (~1,5 KB de embedding por texto distinto, unos
500 MB por día por millón de comentarios), que en ClickHouse se comprime y se puede guardar con
menos precisión (float16) si hace falta.
