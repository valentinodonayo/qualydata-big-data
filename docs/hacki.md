# Texto para cargar el BUIDL en Hacki

Copiá cada bloque en el campo que corresponde. Los números salen de `experimentos/runs.csv`
(práctica, script oficial del jurado) y de `entrega/costo.json` (set final).

## Nombre

Narrativas y cuentas coordinadas

## Descripción corta (una línea)

Qué se dice en 150.000 comentarios, quién lo apoya, desde cuándo y qué cuentas lo empujan. Un comando, 108 segundos y una IA que solo decide lo que las reglas no pueden.

## Track

Qualydata - Big Data

## Repositorio

https://github.com/valentinodonayo/qualydata-big-data

## Sobre el proyecto

### Qué hace

Dejamos de leer comentarios de a uno. El pipeline responde las tres preguntas del desafío con un solo comando: qué narrativas circulan y desde cuándo, quién las apoya y quién las rechaza, y qué cuentas actúan en coordinación.

### Cómo funciona

1. **Agrupa por significado.** Comentarios que dicen lo mismo caen juntos aunque usen otras palabras (embeddings locales, sin servidor).
2. **Descubre las narrativas sin lista previa.** Quien responde discute la afirmación de su padre; esas respuestas unen los grupos.
3. **Arma dos bandos por narrativa.** Las respuestas que se contradicen separan a los que apoyan de los que rechazan.
4. **La IA decide solo lo que hace falta.** Tres modelos votan cuál de los dos bandos afirma, preguntados en los dos órdenes para no caer en el sesgo de posición. Las respuestas quedan guardadas en el repo: el jurado reproduce la entrega sin clave.
5. **Coordinación.** Cuentas que copian textos, llegan juntas a las mismas publicaciones en momentos tranquilos, comparten un hashtag o se responden en círculo. Coincidir en un momento viral casi no cuenta; coincidir de madrugada, mucho.

### Resultados

**Set de práctica, script oficial: 54,72 de 55** (línea base: 16,95). Sin IA: 53,95.

| Medida | Valor |
|---|---|
| Narrativas | 29,99 de 30 |
| Coordinación | 24,72 de 25 |
| Cuentas marcadas | ninguna genuina entre 15.876; 113 de 115 de campaña |
| Acierto de postura con IA | las 13 narrativas |

**Set final (150.000 comentarios, 12.002 cuentas, 10 días):**

- 12 narrativas en tres temas (seguridad, festival, dique). Tres nacen con el período empezado.
- 20,7% de ruido.
- 3 operaciones y 97 cuentas: una empuja que el dique es la solución para el agua; otra, que el festival es un derroche; otra, el cierre de la comisaría del norte.
- 108 s en una Mac. En CPU, sin GPU, corrimos desde cero en 222 s.
- IA: US$ 0,19 a precio de lista para todo el set. Cada re-corrida cuesta US$ 0, porque lee las respuestas guardadas.

### Lo que no funciona todavía

Lo medimos y lo documentamos en el README:

- Si una campaña comenta en los mismos momentos que la gente común, la coordinación marca de más. Lo vimos en un set inventado sin ciclo de día y noche.
- Sin IA, la postura es una moneda al aire en narrativas casi empatadas.
- Una narrativa con menos del 1% de los comentarios puede quedar pegada a otra del mismo tema.

### Probado con otros textos

Lo corrimos sobre recortes del set (10%, 25%, 50%), sobre práctica y final juntos, con todo en mayúsculas y con fechas corridas. Y sobre un set inventado con otros temas (un basural, la tarifa de la luz, una escuela, un club de barrio) con verdad propia: las narrativas se sostienen con B-cubed entre 0,98 y 1,00.

### Reproducible

Un clon limpio, sin ninguna clave, da las mismas narrativas y las mismas cuentas, byte a byte. Semillas fijas, versiones exactas en `uv.lock` y 48 tests.

### Tablero

`reportes/dashboard.html` abre sin internet: la línea de tiempo de cada narrativa, cuánto la apoya y la rechaza, qué operación la empuja y una ficha por operación.

### Cómo correrlo

```bash
make entrega
```

Baja el set final, instala las versiones exactas y escribe `entrega/` y el tablero. Licencia MIT.

## Video de demo

El video (82 segundos, sin audio) se sube como archivo en el campo de video de Hacki.
