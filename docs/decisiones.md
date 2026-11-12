# Decisiones y por qué

Cada decisión se tomó midiendo con el script oficial del jurado sobre el set de práctica
(historial completo en `experimentos/runs.csv`). Puntajes sobre 55 automáticos.

| # | Decisión | Alternativa | Evidencia |
|---|---|---|---|
| 1 | **DuckDB + Parquet** para guardar los datos | Polars, pandas, PostgreSQL | DuckDB y Polars empatan (0,36 s y ~250 MB para 200.000 filas). DuckDB suma SQL, un solo archivo sin servidor y `store_rejects` para filas rotas. Ver `docs/escala.md` |
| 2 | Embeddings **`multilingual-e5-small`** local | Modelos más grandes (bge-m3) o API paga | 72.000 textos en ~42 s en la GPU de la Mac, US$ 0. El vecino más cercano es de la misma narrativa el 96,9% de las veces: el cuello de botella no es el modelo |
| 3 | **Micro-grupos** (UMAP + HDBSCAN) unidos por **votos de los hilos** (Leiden) | Clustering directo a nivel narrativa | Una narrativa se parte en muchos micro-grupos (a favor, en contra, redacción nueva). Las respuestas largas son siempre de la misma narrativa que su padre (100% en práctica): esos votos unen los pedazos sin umbrales a mano. B-cubed 0,986 |
| 4 | **Clasificador de ruido** entrenado con práctica, con rasgos que no dependen del tema | Regla fija de parecido | Ruido F1 de 0,469 a 0,952 (+3,6 puntos). Validado dejando cada tema afuera: F1 0,94 / 0,97 / 0,99. Permitido por el FAQ oficial |
| 5 | **Fecha de origen** = primer momento con ≥5 comentarios del núcleo en 6 h | Mínimo crudo de las fechas | El mínimo crudo depende de un solo comentario: si está mal asignado, el origen se adelanta días. Con el núcleo + ventana, 3 de 3 emergentes dentro de las 3 horas |
| 6 | **Postura por bandos**: las respuestas largas contradicen, así que parten cada narrativa en 2 (corte máximo) | NLI o IA comentario por comentario | Macro-F1 0,916 sin IA y sin costo. La IA solo decide qué bando "apoya" (1 llamada por narrativa) |
| 7 | **Coordinación por evidencia repetida**: pares de cuentas que coinciden en ≥3 publicaciones distintas y ≥30% de su actividad | Marcar ráfagas | Una ráfaga sola es un momento viral (trampa). Precisión 0,98 en práctica. Los tests incluyen las dos trampas (viral y frase copiada) |
| 8 | Capa extra: **comunidades densas en la red de respuestas** (≥1,6 conexiones por cuenta) | Solo texto y tiempo | Atrapa la campaña de textos variados que se responden entre sí (C2). Coordinación de 0,38 a 22,01 puntos con las 4 capas |
| 9 | **Títulos sin IA** = el comentario más típico del bando que apoya | Palabras clave sueltas | "liberada / policia / patrullero" no se entiende; "Llamás a la policía y no viene nadie, es zona liberada" sí |
| 10 | Mantener la **estructura actual** (`pipeline/` con una etapa por archivo) | Reescribir a `src/` + YAML | El pipeline ya puntuaba 51/55; reescribir arriesgaba romperlo a horas del cierre sin sumar puntos |
| 11 | **GPU de la Mac si está, CPU si no** | Forzar CPU siempre | GPU vs CPU en el set final: 99,7% misma narrativa, 99,98% misma postura, 100% mismas cuentas, mismas fechas de origen. Con GPU tarda 108 s y con CPU 141 s |
| 12 | **Orden fijo** en DuckDB y desempate alfabético en frases clave | — | Sin eso, dos corridas daban las mismas filas en otro orden. Ahora 3 corridas salen idénticas byte a byte (también con otra `PYTHONHASHSEED`) |
| 14 | **Capa E de coordinación, "adhesión"**: se suma a una operación quien usa su hashtag propio (≥10 usos, ≥2/3 de ellos de esa operación) o intercambia ≥3 respuestas con ≥2 de sus miembros | Dejar afuera a las cuentas de poca actividad | Práctica: cuentas F1 de 0,901 a 0,932, operaciones de 0,832 a 0,891 (de 51,01 a 51,98), sin perder precisión (0,98). Las 6 cuentas sumadas eran de campaña. Ver la nota sobre el umbral |
| 13 | Dashboard con **una mini línea de tiempo por narrativa** | 12 colores en un gráfico | 12 colores no se distinguen (máximo 8, y menos para daltónicos). La paleta de operaciones se validó con un chequeo automático de daltonismo |

### Nota sobre el umbral del hashtag (transparencia)

La primera versión pedía que el 90% de los usos del hashtag fueran de la operación. Ese 90% fue una
elección arbitraria: en práctica, cualquier valor entre 50% y 94% da exactamente el mismo resultado,
porque los únicos hashtags que hay tienen 94% y 100% de uso por operaciones. Al revisar la salida
del set final (sin respuesta correcta) vimos un hashtag con 76% de uso por una operación y el resto
en 11 cuentas no marcadas que publicaban **las mismas frases plantilla**. Bajamos el umbral a una
"mayoría clara" (2/3), que es una regla general y no menciona ningún hashtag, cuenta ni texto del
set final. En práctica el puntaje no cambió (51,98).

## Qué probamos y descartamos

| Prueba | Resultado | Por qué se descartó |
|---|---|---|
| Adhesión por **sincronía** (comentar cerca de miembros de la operación) | 17.816 cuentas genuinas pasaban el filtro | En publicaciones con mucho movimiento todos comentan cerca de todos |
| Adhesión por **2 respuestas** con miembros | 11 cuentas, solo 6 de campaña | Precisión 55%; con 3 respuestas, 2 de 2 |
| Postura por lista de palabras (línea base oficial) | macro-F1 0,491 | "Es mentira que X" usa las mismas palabras que X |
| Ruido = micro-grupos sin votos, sin clasificador | Ruido F1 0,469 | Muchas raíces cortas de ruido quedaban dentro de micro-grupos |
| Sin filtro de ruido, el mismo estimador de origen | Origen 0,667 (2 de 3) | El ruido mal asignado a una narrativa emergente adelantaba su núcleo |

## Generalización

Ver `docs/generalizacion.md`: tema dejado afuera, primeros 10 días, semillas distintas y datos ensuciados.
