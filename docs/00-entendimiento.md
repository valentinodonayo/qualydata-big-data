# 00 · Entendimiento del desafío (Fase 0)

Escrito el 12/11/2026 a la noche, con el set final ya publicado. Fuente de verdad:
el README y el release `set-final` de https://github.com/qualydata-org/desafio-qualydata.
Si algo de acá contradice al repo oficial, gana el repo.

## 1. Objetivo

Sobre ~150.000 comentarios sintéticos (10 días, 8 fuentes, ~12.000 cuentas) responder:

- **Tarea A, narrativas:** descubrir qué narrativas circulan (sin lista previa), asignar
  cada comentario a una narrativa o a ruido, decir si la **apoya** o la **rechaza** y
  estimar cuándo **nació** cada una.
- **Tarea B, coordinación:** detectar las cuentas que actúan coordinadas y agruparlas
  por operación, sin confundirlas con opinión genuina.

## 2. Reglas (las que nos pueden dejar afuera)

| Regla | Qué implica para nosotros |
|---|---|
| Corre de punta a punta con **un comando** sobre el set final | `correr.py` (y un `make` que lo envuelva). Al podio se lo vuelven a correr. |
| Prohibido etiquetar o editar a mano el set final o la entrega | Nada de narrativas, cuentas, palabras ni fechas escritas a mano. |
| Si el resultado re-corrido "no se parece", quedan afuera | **Semillas fijas** en todo. La IA puede dar diferencias menores (aceptado). |
| Repo **público** con `entrega/` commiteada + video + pitch, cargado en Hacki | Hoy el repo **no tiene remoto**: hay que crearlo en GitHub. |
| README: cómo correr, dónde se guardan los datos y por qué esa base, qué existía antes | Falta escribirlo. |
| Sin claves en el repo; solo datos provistos o públicos | `.env` en `.gitignore`, `.env.example` sin clave real (ya está). |
| Se puede entrenar con la verdad de práctica (FAQ oficial) | El clasificador de ruido (`modelos/ruido.json`) está permitido; tiene que generalizar. |
| Cierre: **viernes 13/11 a las 13:00**. Cuenta el commit clonado a esa hora | Objetivo interno: entrega pusheada a las 08:30. |

## 3. Esquemas de entrada (`datos/<set>/`)

| Archivo | Columnas |
|---|---|
| `fuentes.csv` | `fuente_id, nombre, tipo` |
| `publicaciones.csv.gz` | `publicacion_id, fuente_id, fecha, titular` |
| `comentarios.csv.gz` | `comentario_id, publicacion_id, cuenta_id, fecha, texto, respuesta_a, me_gusta` |
| `cuentas.csv.gz` | `cuenta_id, usuario, fecha_creacion, seguidores, siguiendo` |

Fechas en hora argentina (`-03:00`). `respuesta_a` = id del comentario al que responde, o vacío.
CSV UTF-8 con encabezado, gzip. El set final se verificó con los SHA-256 del release (los 4 coinciden).

| Set | Comentarios | Cuentas | Publicaciones | Días |
|---|---|---|---|---|
| Práctica (con verdad) | 200.000 | 15.991 | 506 | 14 |
| Final (sin verdad) | 150.000 | 12.002 | 372 | 10 (desde 05/10) |

La verdad de práctica tiene 13 narrativas en 3 temas (litio, colectivo, hospital), 3 emergentes,
1 con cambio de redacción desde el día 8, 37.710 comentarios de ruido (19%) y 3 campañas con
115 cuentas (C1 copia casi idéntica con hashtag y cuentas nuevas, C2 cuentas viejas con textos
variados que se responden entre sí, C3 ráfagas nocturnas que originan una narrativa).

## 4. Esquema de salida (`entrega/`)

| Archivo | Columnas / campos |
|---|---|
| `comentarios.csv` | `comentario_id, narrativa` (vacío o `ninguna` = ruido), `postura` (`apoya` / `rechaza`) |
| `narrativas.csv` | `narrativa, titulo, frases_clave` (separadas por `\|`), `fecha_origen` (ISO con zona, ej. `2026-09-07T08:15:00-03:00`) |
| `cuentas.csv` | `cuenta_id, grupo` (vacío si es genuina) |
| `costo.json` | `comentarios_procesados, segundos, tokens_ia, usd_ia, usd_por_mil_comentarios, hardware, almacenamiento, notas` |

Reglas de postura del README: las respuestas cortas ("Tal cual") heredan narrativa y postura
del comentario al que responden; una respuesta que contradice queda en la misma narrativa con
la postura opuesta.

## 5. Puntaje

| Criterio | Pts | Cómo |
|---|---|---|
| Narrativas | 30 | Automático: 50% B-cubed F1 (solo comentarios con narrativa), 15% F1 de ruido, 20% macro-F1 de postura, 15% origen de emergentes (completo a < 3 h, cero a 48 h) |
| Coordinación | 25 | Automático: 70% F1 de cuentas coordinadas, 30% B-cubed F1 por operación |
| Costo, escala y almacenamiento | 20 | Jurado: `costo.json`, tiempo, por qué esa base, cómo escala a millones/día |
| Robustez | 15 | Jurado: un comando, datos sucios, reproducible |
| Claridad y visualización | 10 | Jurado: demo y pitch, "se entiende en un minuto" |

`evaluacion/puntuar.py` es el mismo script del jurado (solo stdlib).

## 6. Puntajes medidos en práctica (12/11, ~20:00)

| Corrida | Narrativas /30 | Coordinación /25 | Total /55 |
|---|---|---|---|
| Línea base oficial (`ejemplos/linea_base.py`) | 12,83 | 4,12 | **16,95** |
| Nuestro pipeline, sin IA (commit `24c5143`) | 29,00 | 22,01 | **51,01** |

Detalle nuestro: B-cubed 0,981 · ruido F1 0,952 · postura macro-F1 0,916 · origen 3/3 ·
cuentas F1 0,901 (P 0,980, R 0,835; 98 marcadas de 115) · grupos B-cubed 0,832.

Primera corrida sobre el **set final** (sin IA): 108 s, 12 narrativas (3 que nacen a mitad del
período: 09/10 y 12/10), 21% ruido, 3 operaciones con 76 cuentas. Los grupos tienen sentido
temático (seguridad, festival, dique, sobreprecios de obra), pero sin verdad no hay puntaje.

## 7. Supuestos y riesgos

1. **Generalización del clasificador de ruido.** Se entrenó con la verdad de práctica usando
   rasgos que no dependen del tema (validación dejando un tema afuera: F1 0,94 a 0,99). Supuesto:
   el ruido del set final tiene la misma "forma" (reacciones, menciones, fuera de tema).
2. **Postura sin IA.** Sin clave, el respaldo es "el bando más grande apoya". En práctica falla
   en 1 de 13 narrativas (B3). En el final no lo podemos medir. Con IA (1 llamada por narrativa,
   ~12 llamadas) debería mejorar y cuesta centavos, pero requiere que Vale cree su `.env`.
3. **Reproducibilidad en otra máquina.** UMAP tiene semilla, pero los embeddings se calcularon en
   la GPU de la Mac (MPS). En la máquina del jurado (CPU u otra GPU) los números pueden variar en
   decimales y mover algún grupo. Hay que medir cuánto cambia la salida CPU vs MPS.
4. **Coordinación:** el recall de cuentas es 0,835 (se escapan ~17 de 115). Las campañas del final
   son otras; los umbrales tienen que salir de los datos (modelo nulo), no de práctica.
5. **El repo no está publicado** (no hay remoto). Sin repo público no hay entrega.
6. **Horario:** quedan ~17 horas al cierre. Prioridad absoluta: entrega válida pusheada.
