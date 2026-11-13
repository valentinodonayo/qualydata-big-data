# QualyData Big Data: narrativas y cuentas coordinadas

Pipeline para el **Desafío QualyData** (Hackathon Argentina Space 2026, Salta). Sobre cientos de
miles de comentarios sociales sintéticos responde dos preguntas:

- **¿Qué narrativas circulan?** Las descubre solo (sin lista previa), asigna cada comentario a una
  narrativa o a ruido, dice si la **apoya** o la **rechaza**, y estima **cuándo nació** cada una.
- **¿Quién las empuja de forma coordinada?** Detecta grupos de cuentas que actúan juntas
  (copian textos, llegan juntas, se responden entre ellas) y los agrupa por operación.

Todo corre local, en una notebook, en menos de dos minutos. La IA es opcional y se usa poco
(6 llamadas por narrativa, no por comentario). Sus respuestas van en una caché versionada, así que
re-correr la entrega **no necesita clave y no gasta nada**.

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
| `make test` | Corre los 48 tests |
| `make limpiar` | Borra la cache (embeddings, UMAP, base). La próxima corrida recalcula todo |

**IA opcional y reproducible.** Para cada narrativa, la IA ayuda a decidir cuál de los dos bandos
"apoya" y le pone un título. Cada pedido sigue este orden:

1. **Caché versionada** (`modelos/cache_ia.json`, va en el repo): si el pedido ya se hizo, se usa esa
   respuesta. Tiene 150 respuestas (78 de práctica y 72 del set final) y guarda huellas de los
   ejemplos, no los textos. **Verificado hoy:** una corrida sin ninguna clave reproduce
   `comentarios.csv`, `narrativas.csv` y `cuentas.csv` byte a byte.
2. **El modelo**, si hay una clave en el entorno o en `.env` (ver `.env.example`). La respuesta se
   agrega a la caché. Sin librerías nuevas (`urllib` estándar).
3. **Sin respuesta**, si no hay ni caché ni clave: deciden las señales sin IA (ver "Postura" más
   abajo). Se avisa en el log.

Proveedores, en orden de preferencia (se usa el primero que tenga clave):

| Proveedor | Clave | Modelos | Pedidos por narrativa |
|---|---|---|---|
| **Vercel AI Gateway** (el que usamos) | `AI_GATEWAY_API_KEY` | Votan tres: `openai/gpt-5-mini`, `openai/gpt-4.1` y `google/gemini-2.5-flash` | 6 (cada modelo, 2 veces) |
| Google Gemini directo | `GEMINI_API_KEY` | `gemini-3.8-flash` | 1 |
| Claude directo | `ANTHROPIC_API_KEY` | `claude-haiku-5-5` | 1 |

Con el gateway, a cada modelo se le pregunta **dos veces**: con los bandos en un orden y en el otro.
Medimos que los modelos tienden a elegir el bando que aparece primero; si un modelo cambia de
respuesta al invertir el orden, se contradice y **no vota** en esa narrativa. Cambiar de modelos es
una variable de entorno: `QUALY_MODELOS_GATEWAY` (por ejemplo `anthropic/claude-haiku-5.5`).

`costo.json` informa los tokens y el gasto de la corrida que generó cada respuesta guardada, a precio
de lista. Con `--sin-ia` el pipeline no consulta ni la caché ni el modelo.

## Resultados

Set de práctica, con el script oficial del jurado (`evaluacion/puntuar.py`):

| Corrida | Narrativas /30 | Coordinación /25 | Total automático /55 |
|---|---|---|---|
| Línea base oficial | 12,83 | 4,12 | 16,95 |
| Este pipeline, sin IA (`--sin-ia`) | 29,23 | 24,72 | 53,95 |
| **Este pipeline, con IA** (como la entrega) | **29,99** | **24,72** | **54,72** |

Detalle en práctica:

| Métrica | Valor |
|---|---|
| Agrupamiento (B-cubed F1) | 1,000 |
| Ruido (F1) | 1,000 |
| Origen de las 3 narrativas emergentes | 1,000 (las 3 dentro de las 3 horas) |
| Postura (macro-F1) | 0,998 con IA · 0,872 sin IA |
| Cuentas coordinadas (F1) | 0,991 (precisión 1,000, cobertura 0,983: 113 de las 115 cuentas de campaña, ninguna genuina marcada) |
| Operaciones (B-cubed F1) | 0,983 |

Evolución: ayer el pipeline sacaba 51,98. Hoy sumaron la capa G de coordinación (52,99), el ruido
por vecinos en hilos (53,22), la IA con votación (53,98), la segunda pasada de la capa G (54,27) y el
filtro de par único más la capa H de tanda de creación (54,63) y el filtro de la red de respuestas (54,72). Historial de cada prueba en
[`experimentos/runs.csv`](experimentos/runs.csv).

**Set final** (`entrega/`, 150.000 comentarios, sin respuesta correcta):

| Qué | Valor |
|---|---|
| Narrativas | 12 (3 nacen con el período empezado: licitación arreglada del dique y cierre de la comisaría el 9/10, grilla del festival el 12/10) |
| Ruido | 20,7% de los comentarios |
| Operaciones coordinadas | 3, con 97 cuentas (50 / 25 / 22) |
| Tiempo desde cero | 108 s |
| IA | 72 respuestas, 117.915 tokens, US$ 0,187 a precio de lista |

**¿Generaliza?** Ver [`docs/generalizacion.md`](docs/generalizacion.md): con un tema que el pipeline
nunca vio, con solo 10 días, con otras semillas y con datos ensuciados a propósito. Agrupamiento y
ruido se sostienen en los 9 experimentos; la postura sin IA es la parte frágil.

## Dashboard

El pipeline genera solo [`reportes/dashboard.html`](reportes/dashboard.html): un archivo, sin internet
ni servidor (descargalo y abrilo con el navegador). Responde en un minuto **qué circula** (cada
narrativa con su título), **desde cuándo** (una mini línea de tiempo por narrativa con la fecha de
origen marcada) y **quién lo empuja** (cada operación aparece solo en la narrativa que empuja, más la
red de cuentas coordinadas con tarjetas que explican cada operación). Funciona en celular y en modo
oscuro.

## Cómo funciona

```
CSV.gz ──► DuckDB + Parquet ──► embeddings locales ──► narrativas y ruido ──► coordinación ──► postura ──► entrega/
           (limpieza y         (multilingual-e5-small, (UMAP+HDBSCAN         (grafo de cuentas  (2 bandos por hilos;
            reporte de calidad)  1 vez por texto distinto) + hilos + Leiden)    + Leiden)          IA + señales deciden cuál apoya)
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
4. **Ruido** (paso 5 de [`narrativas.py`](pipeline/narrativas.py)): la idea es que el ruido ("Feliz
   cumple", "@fulano mirá") nunca entra en una discusión, y los textos que se le parecen tampoco.
   - Un comentario que está en un hilo (responde o le responden) **nunca** es ruido.
   - Un comentario suelto es ruido si, entre sus ~50 comentarios más parecidos (por embeddings; cada
     texto cuenta por sus copias), casi ninguno participa de un hilo: menos del 10% de la tasa
     normal del set. Los sueltos sin palabras (solo emojis o vacíos) también son ruido.
   - Lo que no es ruido y quedó sin narrativa se rescata: toma la narrativa de sus copias que están
     en hilos o, si no tiene, la del micro-grupo más parecido.

   No usa etiquetas ni depende del tema. En práctica: 0 comentarios de narrativa marcados como ruido
   (antes 3.085) y 0 ruido sin detectar (antes 630). El clasificador entrenado (`modelos/ruido.json`)
   queda **solo como respaldo**, para sets donde menos del 2% de los comentarios está en hilos.
5. **Fecha de origen**: el primer momento en que el **núcleo** de la narrativa (la mitad más parecida
   al centro) junta 5 comentarios en 6 horas. Un comentario suelto mal asignado no la adelanta.
6. **Coordinación** ([`coordinacion.py`](pipeline/coordinacion.py)): corre antes que la postura,
   porque la postura usa de qué lado escriben las cuentas coordinadas. Todo es conteo y grafos, sin IA.
   - **Pares que repiten** (capas A, B y C): mismo texto en 30 min, misma publicación en la misma
     franja de 15 min cuando hay poca gente, o respuestas entre ellas. Un par queda unido si coincide
     en al menos 3 publicaciones distintas. Así distingue una operación de un momento viral (mucha
     gente, una sola vez).
   - **Red de respuestas** (capa D): comunidades de "quién le responde a quién" con muchos más
     ciclos que una charla normal. De cada comunidad quedan solo las cuentas que intercambiaron al
     menos 2 respuestas adentro: una sola respuesta con un miembro es charla casual.
   - **Adhesión** (capa E): suma a una operación las cuentas de poca actividad que usan su
     **hashtag propio** o que intercambian varias respuestas con sus miembros.
   - **Llegadas raras** (capa G, nueva): suma la cuenta que llega a las mismas publicaciones que los
     miembros de una operación en momentos tranquilos. Cada coincidencia en una franja de 15 min
     pesa 1/(cuentas presentes − 1): con poca gente pesa mucho, en un momento viral casi nada.
     Umbrales: puntaje ≥ 1,4, densidad ≥ 0,03 por publicación de la cuenta y ≥ 2 publicaciones
     distintas en común. Se repite hasta 3 pasadas, porque una cuenta recién sumada puede delatar a
     otra. En práctica recuperó 7 de las 9 cuentas de la campaña "de cobertura" que se escapaban.
   - **Filtro de par único**: una cuenta que entra por un solo par fuerte, y ese par es menos del
     10% de su propia actividad, se suelta. Saca a las genuinas muy activas que coinciden por volumen.
   - **Tanda de creación** (capa H, nueva): usa la metadata de las cuentas. Si casi todos los
     miembros de una operación se crearon en una ventana de ≤ 10 días y ≥ 80% comparte la forma del
     usuario (por ejemplo palabra + 4 dígitos), se suma la cuenta no marcada creada en esa ventana
     (±1 día) con esa misma forma. Nada de fechas ni nombres escritos a mano: todo es relativo a cada
     operación ya detectada. En el set final sumó 4 cuentas.
   - **Tope a las capas de suma** (E, G y H): si una de esas capas propone más cuentas que las que
     la operación ya tiene, la señal dejó de ser específica y no se suma ninguna. Es un seguro para
     sets con otra dinámica (ver "Probado con otros textos").
7. **Postura** ([`postura.py`](pipeline/postura.py), [`ia.py`](pipeline/ia.py)): dentro de cada
   narrativa, las respuestas largas contradicen a su padre, así que parten los micro-grupos en 2
   bandos (corte máximo). Las respuestas cortas ("Tal cual") heredan la postura del padre. Falta
   saber cuál de los dos bandos "apoya":
   - El modelo ve ejemplos variados de cada bando (una frase por micro-grupo, o sea por cada forma
     distinta de decirlo).
   - Si **2 o más modelos votan y todos coinciden**, gana la IA.
   - Si no, se suman puntos por bando: 1 por cada modelo que lo votó; 2 por el lado de la operación
     coordinada, cuando ≥ 80% de sus comentarios cae en un bando; y al bando más grande 1,5 si tiene
     ≥ 54% de la narrativa o 0,5 si no.
   - Sin caché ni clave no hay votos y decide esa misma suma (operación y tamaño). Con `--sin-ia` se
     usa directamente el bando más grande.

   En práctica la regla acierta la polaridad de 13 de 13 narrativas. En el set final la IA coincide
   con el bando más grande en las 12 (en 2 se abstiene y decide el tamaño, 60-63%).
8. **Salida y costo** ([`salida.py`](pipeline/salida.py), [`costo.py`](pipeline/costo.py)): escribe
   los 4 archivos con el formato exacto y arma `costo.json` midiendo cada etapa. Si una etapa falla,
   se anota en el log y la entrega se escribe igual.

## Costo y escala

Números del set final, tomados de [`entrega/costo.json`](entrega/costo.json):

| Concepto | Valor |
|---|---|
| Tiempo desde cero | **108 s** en una notebook Apple M5 (embeddings en la GPU integrada) |
| IA, la corrida que llenó la caché | 72 respuestas, 117.915 tokens, **US$ 0,187** a precio de lista |
| IA por cada 1.000 comentarios | US$ 0,0012 |
| IA en cada re-corrida | **US$ 0** (lee la caché) |

El costo de IA **no crece con el volumen**: son 6 llamadas por narrativa, no por comentario. El
cómputo sale ≈US$ 0,0001 por cada 1.000 comentarios (con un precio de VM supuesto). Cómo llevarlo a
millones de comentarios por día: [`docs/escala.md`](docs/escala.md). Las decisiones de diseño, con la
evidencia de cada una: [`docs/decisiones.md`](docs/decisiones.md).

## Dónde se guardan los datos y por qué DuckDB

- **DuckDB** (`cache/qualydata.duckdb`): una base en un solo archivo, sin servidor. Medido contra
  Polars y pandas (`evaluacion/medir_almacenamiento.py`): empata con Polars en velocidad (0,36 s para
  200.000 comentarios) y gana en lo demás. El jurado la re-crea con un comando, sin instalar nada
  aparte. Es columnar, hecha para contar y agrupar millones de filas, y lee los `.csv.gz` directo
  apartando las filas rotas.
- **Parquet** (`cache/parquet/`): copia limpia en el formato estándar comprimido, que abre cualquier
  herramienta (pandas, Spark, BigQuery, ClickHouse).
- **Cache** (`cache/embeddings/`, `cache/umap/`): los cálculos caros se guardan con una huella del
  contenido. Si los datos no cambian, no se recalculan.

## Reproducibilidad y robustez

- Versiones exactas en `uv.lock` (`uv sync --frozen`), Python 3.12 en `.python-version` y el modelo
  de embeddings fijado por revisión.
- Semilla fija (42) en UMAP, Leiden y todo muestreo, y orden fijo en las consultas.
- **IA reproducible:** las respuestas están en `modelos/cache_ia.json`. Verificado hoy: sin ninguna
  clave, la corrida reproduce `comentarios.csv`, `narrativas.csv` y `cuentas.csv` byte a byte.
- **Otra máquina:** si no hay GPU de Mac, usa CPU (más lento). Se puede forzar con
  `QUALY_DISPOSITIVO=cpu`. Los embeddings pueden cambiar en los últimos decimales; si por eso cambia
  algún ejemplo, la caché reconoce cada narrativa por los ejemplos en común (ver "Límites conocidos").
  **Verificado el 13/11 con un clon limpio, solo CPU y sin ninguna clave** (`make entrega`): termina en
  222 s, `narrativas.csv` y `cuentas.csv` salen idénticos byte a byte, y en `comentarios.csv` el 99,94%
  de los comentarios queda en la misma narrativa y el 99,99% con la misma postura. Las 72 respuestas
  de la IA salieron de la caché.
- **Datos sucios:** columnas desordenadas o faltantes, columnas extra, filas rotas, ids repetidos,
  fechas inválidas o con otra zona horaria, textos vacíos, respuestas a comentarios que no existen y
  cuentas sin metadata. Todo queda contado en `cache/reportes/calidad.json`.
- **Nunca se queda sin entrega:** si una etapa falla, se anota en `cache/reportes/log.txt`, se sigue
  con las demás y al final se valida el formato de los 4 archivos (`pipeline/validar.py`).
- **Tests:** `make test` corre 48 tests, incluida una prueba de punta a punta sobre una muestra
  ensuciada, casos trampa de coordinación (momento viral y frase copiada sin coordinación), el ruido
  por vecinos y la votación de la IA (sin llamar a ningún modelo).

## Probado con otros textos

Además del hold-out, el 13/11 se probó el pipeline con datos distintos a los de práctica. Todo sin
IA y puntuado con el script oficial; los scripts quedan en el repo para repetirlo.

**Recortes y mezclas de los sets oficiales** (`evaluacion/estres.py`):

| Prueba | Narrativas encontradas / reales | B-cubed | Ruido F1 | Coordinación |
|---|---|---|---|---|
| 50% de las publicaciones | 13 / 13 | 0,994 | 1,000 | precisión 0,97, cobertura 0,58 |
| 25% | 13 / 13 | 0,996 | 1,000 | precisión 1,00, cobertura 0,63 |
| 10% (25.000 comentarios) | 13 / 13 | 0,994 | 1,000 | no marca a nadie |
| Práctica + final juntos (350.000) | 25 / 25 | 0,987 | 1,000 | 6 operaciones, sin mezclar cuentas de los dos sets |
| Sin ninguna campaña | 13 / 13 | 0,996 | 1,000 | 0 cuentas marcadas (correcto) |
| Fechas corridas 400 días y en UTC | salida idéntica, con los orígenes corridos 400 días | | | |
| Todo en mayúsculas, o con un emoji al final | salida idéntica a la del mismo recorte | | | |

De esas pruebas salieron cuatro arreglos, que no cambian la práctica ni el set final: un archivo
secundario roto (por ejemplo `cuentas.csv.gz` de 0 bytes) ya no deja la entrega vacía; en sets muy
grandes se separan dos narrativas que quedaban pegadas; si el set no tiene hilos se agrupa por
parecido en vez de entregar todo como ruido; y en sets de menos de 20.000 comentarios el ruido usa
menos vecinos.

**Un set inventado con otros temas** (`evaluacion/generar_set.py`, `evaluacion/medir_inventado.py`):
60.000 comentarios sobre un basural, la tarifa de la luz, una escuela y un club de barrio, con 10
narrativas, 3 campañas y verdad propia, más ocho variantes que rompen de a una las reglas del set
oficial.

| Variante | Narrativas | B-cubed | Ruido F1 | Origen |
|---|---|---|---|---|
| Mismas reglas que el oficial, texto nuevo | 10 / 10 | 0,997 | 1,000 | 1,000 |
| 15% de las respuestas largas acuerdan con su padre | 10 / 10 | 0,997 | 1,000 | 1,000 |
| 5% de las respuestas cruzan a otra narrativa | 10 / 10 | 0,982 | 1,000 | 1,000 |
| Narrativas con mucho vocabulario compartido | 10 / 10 | 0,996 | 1,000 | 1,000 |
| Pocos hilos (8% de respuestas) | 10 / 10 | 0,995 | 0,996 | 1,000 |
| Una narrativa muy chica (0,9% del set) y una gigante | 9 / 10 | 0,980 | 1,000 | 0,667 |
| El ruido a veces recibe respuestas | 10 / 10 | 0,997 | 0,175 | 1,000 |

Lo que dicen estas pruebas: **el agrupamiento en narrativas se sostiene con temas y textos nuevos**.
Lo que se rompe queda en "Límites conocidos".

## Código previo al evento

Todo el código de `pipeline/`, `correr.py`, `entrenar_ruido.py`, `evaluacion/evaluar.py` y los tests
se escribió durante el hackathon (11 al 13/11/2026), con asistencia de Claude Code. Venían del repo
oficial y no se modificaron: `ejemplos/linea_base.py` (línea base) y `evaluacion/puntuar.py` (script
del jurado). Librerías y modelos públicos usados: DuckDB, pandas, NumPy, PyArrow, scikit-learn,
sentence-transformers con `intfloat/multilingual-e5-small` (MIT), umap-learn, hdbscan, igraph y
leidenalg. Servicios opcionales de IA, todos por API y sin que el pipeline dependa de ellos:
**Vercel AI Gateway** (con modelos de OpenAI y Google; es el que generó la caché), la API REST de
Google Gemini (sin SDK) y el SDK de Anthropic.

## Límites conocidos

- **Polaridad en narrativas "positivas" parejas.** Cuando la narrativa es una promesa o una propuesta
  y los dos bandos tienen casi el mismo tamaño, los modelos a veces se contradicen al invertir el
  orden y no votan. Ahí decide el tamaño, que con bandos parejos es una señal débil. Sin IA es peor:
  en narrativas con 49,5% a 50,5% de apoyo, "el bando más grande" es una moneda al aire (postura
  macro-F1 0,872 sin IA contra 0,998 con IA en práctica).
- **Coordinación: quedan 2 cuentas sin marcar en práctica (ninguna genuina marcada).** Las 2 son de la
  campaña "de cobertura" (cuentas viejas, textos variados) y coinciden con la operación tan poco como
  una cuenta genuina. Se probaron 7 enfoques (incluidos NCD por compresión y ráfagas compartidas, de
  la literatura 2025): ninguno las recupera sin marcar genuinas
  ([`reportes/mejoras-13-11.md`](reportes/mejoras-13-11.md)).
- **La coordinación depende de la dinámica del set.** Los umbrales son absolutos (por ejemplo, 3
  publicaciones en común). Con la mitad de las publicaciones la cobertura baja a 0,58 y con el 10% no
  se marca a nadie. Y en el set inventado (variante con respuestas que acuerdan), donde las campañas comentan en los mismos momentos que la
  gente común (sin ciclo de día y noche), marca de más: con el tope a las capas de suma pasó de 3.906
  a 511 cuentas marcadas sobre 115 reales, que sigue siendo malo. En los dos sets oficiales las
  campañas operan en momentos tranquilos y la precisión es de 0,98 a 1,00.
- **El ruido supone que nadie le responde al ruido.** En los dos sets oficiales se cumple exacto. Si
  el 10% del ruido estuviera en hilos, el F1 de ruido caería a 0,18.
- **Narrativas muy chicas.** Una narrativa con menos del 1% de los comentarios puede quedar pegada a
  otra del mismo tema (la más chica de práctica tiene 1,75%).
- **Sin hilos no hay narrativas finas.** Con `respuesta_a` vacío el respaldo agrupa por parecido y
  junta bien por tema, pero mezcla narrativas del mismo tema (B-cubed 0,53).
- **Reconocimiento "difuso" de la caché de IA.** Si en otra máquina los grupos cambian un poco, la
  caché reconoce cada narrativa por los ejemplos en común (pide al menos 8 entre los dos bandos). Si
  cambian mucho, no la reconoce y esa narrativa se decide sin votos de IA (operación y tamaño).
- **`usd_ia` es precio de lista.** La cuenta del gateway está en la capa gratis: el número de
  `costo.json` es lo que costaría pagarlo, no lo que se pagó.

## Estructura

```
correr.py                 comando único del pipeline
pipeline/                 una etapa por archivo
modelos/cache_ia.json     respuestas de la IA (caché versionada: reproduce la entrega sin clave)
modelos/ruido.json        clasificador de ruido de respaldo (coeficientes)
entrenar_ruido.py         cómo se entrenó ese clasificador
evaluacion/               script oficial del jurado + registro de pruebas + hold-out
experimentos/runs.csv     historial de puntajes
experimentos/generalizacion.csv  resultados del hold-out
docs/                     entendimiento, decisiones, generalización y escala
docs/hacki.md             texto para la plataforma del hackathon
reportes/                 diagnóstico de puntos perdidos y registro de mejoras (aceptadas y descartadas)
reportes/dashboard.html   dashboard generado por el pipeline sobre el set final
tests/                    tests (make test)
entrega/                  resultado sobre el set final (lo que se puntúa)
datos/                    sets de datos (no se suben al repo)
```

Licencia MIT. Todos los datos son sintéticos.
