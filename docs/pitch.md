# Guion del video pitch (≈ 80 segundos)

El video de `reportes/demo.mp4` tiene las placas y el recorrido del tablero, pero no voz. Este
guion es para grabar la voz encima. Cada bloque tiene su tiempo en el video: leelo sin apuro y
pausá al cambiar de placa.

## Lo que dice cada escena

**0:00 · Placa 01 (6 s)**
"Hay 150.000 comentarios en diez días. Nadie los puede leer entero. Nosotros armamos algo que sí."

**0:06 · Placa 02 (7 s)**
"Es un solo comando, `make entrega`. En 108 segundos, en una notebook, responde qué se dice, quién
lo apoya y quién lo rechaza, desde cuándo, y qué cuentas lo empujan de forma armada."

**0:13 · Placa 03 (9 s)**
"Agrupamos por significado, no por palabras. Lo que se contesta discute lo mismo. Dentro de cada
narrativa, las respuestas que se contradicen arman dos bandos. Y la IA solo decide cuál de los dos
afirma: tres modelos votan, cada uno preguntado en dos órdenes."

**0:22 · Tablero (36 s, paneo lento)**
"Esto es lo que sale. Cada narrativa con su título, cuándo nació y cuánto apoyo tiene. Abajo, las
operaciones: algunas tienen cuentas creadas la misma semana, y otras son cuentas viejas que comentan
de cobertura. Cada operación empuja una sola narrativa."

**0:58 · Placa 04 (9 s)**
"En el set de práctica, con el script oficial del jurado: 54,72 de 55. Sin IA, 53,95. Y ninguna
cuenta genuina marcada."

**1:07 · Placa 05 (8 s)**
"Lo que falla también lo medimos. Si una campaña comenta en los mismos momentos que la gente común,
la coordinación marca de más. Y sin IA, la postura es una moneda al aire en narrativas casi
empatadas."

**1:15 · Placa 06 (7 s)**
"Todo está en el repo. Un clon limpio, sin ninguna clave, da la misma entrega. Gracias."

## Cómo se compara con los otros dos

Lo que está cargado en Hacki hoy:

| | Radar de narrativas | INTI-GRAPH | Nosotros (hoy) | Nosotros (propuesto) |
|---|---|---|---|---|
| Gancho | "Qué narrativas circulan… con un comando y US$ 0,012 de IA" | "… en un comando, sin GPU y con USD 0 de IA" | Empieza con "Toma el set del desafío…" | Empieza con la pregunta del jurado |
| Costo | US$ 0,012 | USD 0 | US$ 0,19 a precio de lista | Lo decimos igual, con el "US$ 0 al re-correr" |
| Números | Tabla simple | Tabla con puntaje | Tablas con B-cubed y macro-F1 | Tres números: 54,72, 53,95 y "ninguna genuina" |
| Honestidad | Sin límites declarados | Calibrado con práctica (lo dicen) | Límites en el README | Límites en el texto, en una sección propia |
| Jerga | Baja | Media | Alta ("bit", "B-cubed") | Baja |
| Largo | Corto | Medio | Largo | Medio |

Lo que tenemos y ellos no: probamos con un set inventado de otros temas, medimos el
recorte del 10%, y publicamos lo que falla. Eso es lo que hay que destacar en el pitch, no un
número más alto.

## Consejos para grabarlo

- Grabá la voz con QuickTime (Archivo → Nueva grabación de audio) mientras se reproduce el video, o
  grabala aparte y unila con ffmpeg.
- Si el paneo del tablero te gana de velocidad, bajalo: podés cortar la voz en dos frases.
- No digas "IA" tres veces seguidas: dale espacio al resto del pitch.
- Si te piden la versión sin números, leé solo la primera frase de cada escena.
