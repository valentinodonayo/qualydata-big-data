# Costo y escala: de 150.000 comentarios a millones por día

Todos los números de este documento salen de mediciones propias en una MacBook (Apple M5,
10 núcleos, 16 GB de RAM), salvo donde dice **supuesto**. Los scripts que los generan están
en `evaluacion/`.

## 1. Lo que cuesta hoy (set final, 150.000 comentarios)

| Etapa | Segundos (GPU de la Mac) | Segundos (solo CPU) | Qué hace |
|---|---|---|---|
| Datos | 0,6 | 0,6 | Leer CSV.gz, limpiar, guardar DuckDB + Parquet |
| Embeddings | 35 | 130 | 49.677 textos distintos (de 150.000 comentarios) |
| Narrativas | 71 | 63 | UMAP + HDBSCAN + votos de hilos + Leiden + ruido |
| Postura | 0,6 | 0,4 | 2 bandos por narrativa (grafo de respuestas) |
| Coordinación | 4,1 | 3,4 | Pares de cuentas + red de respuestas + hashtag propio |
| **Total** | **≈111** | **≈141** | Con cache: ≈20 s. Entre corridas varía de 108 a 120 s |

- **IA:** 0 tokens y US$ 0 en la entrega (corre sin IA). Con IA serían ~12 llamadas a Claude Haiku
  (una por narrativa, ~1.000 tokens de entrada y ~150 de salida cada una): unos US$ 0,002 por corrida,
  o sea **US$ 0,000014 por cada 1.000 comentarios** (precio por token en `pipeline/ia.py`).
- **Cómputo:** ≈0,74 ms por comentario con GPU y ≈0,94 ms solo con CPU. Con una VM de 8 a 10 núcleos
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
| Coordinación (pares) | Cuadrático **dentro** de cada franja de 15 min | Acotado: las franjas con >30 cuentas se ignoran |
| Coordinación (red de respuestas) | Lineal en respuestas | Ninguno |

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
6. **IA:** sigue siendo 1 llamada por narrativa **nueva**, no por comentario. El gasto de IA no
   depende del volumen sino de cuántas narrativas aparecen.

### Cuánto costaría (con nuestras mediciones)

Para **1 millón de comentarios por día**, con ~33% de textos distintos (lo medido en los dos sets):

| Concepto | Cálculo | Por día |
|---|---|---|
| Embeddings en CPU | 330.000 textos × 2,6 ms | ≈14 min de una VM |
| Asignación ANN | 1.000.000 × 0,27 µs | < 1 s |
| Re-agrupado horario | ~10% sin narrativa = 100.000/día, en 24 lotes de ~4.000 | ≈5 min de una VM (**estimado**: hoy agrupar 50.000 textos tarda 63 a 71 s; se toma un margen amplio) |
| Coordinación | lineal, ≈4 s por cada 150.000 | ≈27 s |
| IA | ~10 narrativas nuevas × US$ 0,0002 | ≈US$ 0,002 |
| **Total cómputo** | ≈20 min de una VM de 8 a 10 núcleos a US$ 0,30/h (**supuesto**) | **≈US$ 0,10** |

Eso da **≈US$ 0,0001 por cada 1.000 comentarios**: una sola VM chica alcanza para varios millones
por día. Lo que más crece es el almacenamiento (~1,5 KB de embedding por texto distinto, unos
500 MB por día por millón de comentarios), que en ClickHouse se comprime y se puede guardar con
menos precisión (float16) si hace falta.
