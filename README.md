# QualyData Big Data: narrativas y cuentas coordinadas

Pipeline para el **Desafío QualyData** (Hackathon Argentina Space 2026, Salta). Sobre cientos de
miles de comentarios sociales sintéticos responde dos preguntas:

- **¿Qué narrativas circulan?** Las descubre solo (sin lista previa), asigna cada comentario a una
  narrativa o a ruido, dice si la **apoya** o la **rechaza**, y estima **cuándo nació** cada una.
- **¿Quién las empuja de forma coordinada?** Detecta grupos de cuentas que actúan juntas
  (copian textos, llegan juntas, se responden entre ellas) y los agrupa por operación.

Todo corre local, en una notebook, en un par de minutos y **sin gastar en IA** (la IA es opcional).

## Cómo correrlo (un comando)

Requisitos: [uv](https://docs.astral.sh/uv/) (instala Python 3.12 y las librerías solo) y conexión a
internet la primera vez (baja el set final y el modelo de embeddings, ~120 MB).

```bash
make entrega
```

Eso baja el set final del release oficial a `datos/final/` (si no está), instala las versiones
exactas de `uv.lock`, corre el pipeline y escribe los 4 archivos en `entrega/`. Sin `make`:

```bash
uv sync --frozen && uv run python correr.py --datos datos/final --entrega entrega
```

Otros comandos:

| Comando | Qué hace |
|---|---|
| `make practica` | Corre sobre el set de práctica y lo puntúa con el script oficial del jurado |
| `make test` | Corre los tests |
| `make limpiar` | Borra la cache (embeddings, UMAP, base). La próxima corrida recalcula todo |

**IA opcional.** Si existe la variable `ANTHROPIC_API_KEY` (en el entorno o en un archivo `.env`,
ver `.env.example`), se hace 1 llamada a Claude Haiku por narrativa (~12 en total) para decidir qué
bando "apoya" y ponerle un título legible. Sin clave, el pipeline usa un respaldo sin IA y anda igual.
La entrega de este repo se generó **sin IA** (`usd_ia: 0`).

## Resultados

| Corrida | Narrativas /30 | Coordinación /25 | Total automático /55 |
|---|---|---|---|
| Línea base oficial, set de práctica | 12,83 | 4,12 | 16,95 |
| **Este pipeline, set de práctica** (sin IA) | **29,00** | **22,01** | **51,01** |

Detalle en práctica: agrupamiento B-cubed F1 0,981 · ruido F1 0,952 · postura macro-F1 0,916 ·
origen de las 3 narrativas emergentes dentro de las 3 horas · cuentas coordinadas F1 0,901
(precisión 0,98) · operaciones B-cubed F1 0,832. Historial de cada prueba en
[`experimentos/runs.csv`](experimentos/runs.csv).

En el **set final** (150.000 comentarios, sin respuesta correcta) encuentra 12 narrativas, 3 de ellas
nacidas a mitad del período, 21% de ruido y 3 operaciones coordinadas.

## Cómo funciona

```
CSV.gz ──► DuckDB + Parquet ──► embeddings locales ──► narrativas ──► postura ──► coordinación ──► entrega/
           (limpieza y         (multilingual-e5-small, (UMAP+HDBSCAN  (2 bandos   (grafo de cuentas
            reporte de calidad)  1 vez por texto distinto) + hilos + Leiden) por hilos)  + Leiden)
```

1. **Datos** ([`pipeline/datos.py`](pipeline/datos.py)): lee los CSV comprimidos con DuckDB, aparta
   las filas rotas en vez de cortarse, saca duplicados, convierte fechas con zona horaria y deja un
   reporte de calidad en `cache/reportes/calidad.json`.
2. **Texto y embeddings** ([`texto.py`](pipeline/texto.py), [`embeddings.py`](pipeline/embeddings.py)):
   normaliza (minúsculas, menciones, links, hashtags, letras repetidas, emojis) y convierte cada texto
   **distinto** en un vector de significado con un modelo multilingüe chico que corre local. Los
   textos repetidos se calculan una sola vez y todo queda en cache.
3. **Narrativas** ([`narrativas.py`](pipeline/narrativas.py)): UMAP + HDBSCAN arman micro-grupos muy
   puros (cada uno, una forma de decir algo). Después, las **respuestas** votan: si un comentario de
   un micro-grupo le responde a otro, los dos son de la misma discusión. Leiden une los micro-grupos
   con esos votos, así una narrativa que cambia de redacción con los días sigue unida. No hay que
   decirle cuántas narrativas hay.
4. **Ruido**: los micro-grupos que nadie discute son ruido, y un clasificador chico (entrenado con la
   verdad de práctica, con rasgos que **no dependen del tema**) limpia las raíces cortas dudosas.
5. **Fecha de origen**: el primer momento en que el **núcleo** de la narrativa (la mitad más parecida
   al centro) junta 5 comentarios en 6 horas. Un comentario suelto mal asignado no la adelanta.
6. **Postura** ([`postura.py`](pipeline/postura.py)): dentro de cada narrativa, las respuestas largas
   contradicen a su padre, así que parten los micro-grupos en 2 bandos (corte máximo). Las respuestas
   cortas ("Tal cual") heredan la postura del padre. Qué bando "apoya" lo decide la IA o, sin IA, el
   bando más grande.
7. **Coordinación** ([`coordinacion.py`](pipeline/coordinacion.py)): une pares de cuentas que
   **repiten** coincidencias (mismo texto en 30 min, misma publicación en la misma franja de 15 min
   cuando hay poca gente, respuestas entre ellas) en al menos 3 publicaciones distintas. Aparte,
   busca comunidades en la red de "quién le responde a quién" que tienen muchos más ciclos que una
   charla normal. Así distingue una operación de un momento viral (mucha gente, una sola vez).
8. **Salida y costo** ([`salida.py`](pipeline/salida.py), [`costo.py`](pipeline/costo.py)): escribe
   los 4 archivos con el formato exacto y arma `costo.json` midiendo cada etapa. Si una etapa falla,
   se anota en el log y la entrega se escribe igual.

## Dónde se guardan los datos y por qué DuckDB

- **DuckDB** (`cache/qualydata.duckdb`): una base en un solo archivo, sin servidor. El jurado la
  re-crea con un comando, sin instalar nada aparte. Es columnar, hecha para contar y agrupar millones
  de filas, y lee los `.csv.gz` directo apartando las filas rotas.
- **Parquet** (`cache/parquet/`): copia limpia en el formato estándar comprimido, que abre cualquier
  herramienta (pandas, Spark, BigQuery, ClickHouse).
- **Cache** (`cache/embeddings/`, `cache/umap/`): los cálculos caros se guardan con una huella del
  contenido. Si los datos no cambian, no se recalculan.

## Reproducibilidad

- Versiones exactas en `uv.lock` (`uv sync --frozen`) y Python 3.12 fijado en `.python-version`.
- Semilla fija (42) en UMAP, Leiden y todo muestreo.
- Sin IA, dos corridas dan la misma salida. Con IA puede variar algún título.

## Código previo al evento

Todo el código de `pipeline/`, `correr.py`, `entrenar_ruido.py`, `evaluacion/evaluar.py` y los tests
se escribió durante el hackathon (11 al 13/11/2026), con asistencia de Claude Code. Venían del repo
oficial y no se modificaron: `ejemplos/linea_base.py` (línea base) y `evaluacion/puntuar.py` (script
del jurado). Librerías y modelos públicos usados: DuckDB, pandas, NumPy, PyArrow, scikit-learn,
sentence-transformers con `intfloat/multilingual-e5-small` (MIT), umap-learn, hdbscan, igraph,
leidenalg y el SDK de Anthropic (opcional).

## Límites conocidos

- Sin IA, la regla "el bando más grande apoya" se equivoca en 1 de las 13 narrativas de práctica.
- El clasificador de ruido aprendió de la práctica. Usa rasgos independientes del tema y se validó
  dejando cada tema afuera (F1 0,94 a 0,99), pero un ruido de forma muy distinta podría colarse.
- La coordinación detecta el 84% de las cuentas de campaña en práctica con 98% de precisión. Las
  campañas más sutiles (pocas cuentas, textos variados, sin responderse) pueden escaparse.

## Estructura

```
correr.py              comando único del pipeline
pipeline/              una etapa por archivo
modelos/ruido.json     clasificador de ruido (coeficientes)
entrenar_ruido.py      cómo se entrenó ese clasificador
evaluacion/            script oficial del jurado + registro de pruebas
experimentos/runs.csv  historial de puntajes
docs/                  entendimiento del desafío, decisiones y escala
entrega/               resultado sobre el set final (lo que se puntúa)
datos/                 sets de datos (no se suben al repo)
```

Licencia MIT. Todos los datos son sintéticos.
