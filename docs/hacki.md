# Texto para cargar el BUIDL en Hacki

> Copiá cada bloque en el campo que corresponde. Los números salen de `experimentos/runs.csv`
> (práctica, script oficial del jurado) y de `entrega/costo.json` (set final).

## Nombre

Narrativas y cuentas coordinadas

## Descripción corta (una línea)

Qué se dice, quién lo apoya, desde cuándo y qué cuentas lo empujan: 150.000 comentarios en un comando y 108 segundos, con la IA usada solo donde hace falta.

## Track

Qualydata - Big Data

## Repositorio

https://github.com/valentinodonayo/qualydata-big-data

## Sobre el proyecto

### Qué hace

Toma el set del desafío y responde, sin intervención manual, las dos preguntas del jurado: **qué narrativas circulan** (qué comentario pertenece a cuál, quién la apoya, quién la rechaza y cuándo nació) y **qué parte de la conversación está armada** (qué cuentas actúan coordinadas y en qué operación). Corre de punta a punta con `make entrega`, en una notebook, sin servicios externos.

### Cómo funciona

- **Datos.** DuckDB lee y limpia los CSV (fechas rotas, filas duplicadas, columnas desordenadas) y deja una copia en Parquet. Sin servidor.
- **Significado.** Embeddings multilingües locales (`multilingual-e5-small`) agrupan los comentarios por lo que dicen, no por las palabras que usan.
- **Narrativas.** Cientos de micro-grupos muy puros se unen cuando sus comentarios se responden entre sí: una respuesta discute dentro de la narrativa de su padre, aunque la rechace. No hay lista previa ni cantidad fija.
- **Ruido.** Un comentario suelto es ruido cuando casi ninguno de sus 50 comentarios más parecidos participa de una discusión. No usa etiquetas ni depende del tema.
- **Postura.** Dentro de cada narrativa, las respuestas que se contradicen separan dos bandos. Lo único que necesita "entender" es cuál de los dos afirma: un bit por narrativa.
- **IA solo para ese bit.** Tres modelos (por Vercel AI Gateway) miran ejemplos de cada bando y votan cuál afirma; a cada uno se le pregunta dos veces, con los bandos en un orden y en el otro, porque tienden a elegir el que aparece primero. Si se contradicen, decide la estructura (tamaño de los bandos y lado de la operación coordinada). Las respuestas quedan guardadas en el repositorio: el jurado reproduce la entrega sin clave y sin internet.
- **Coordinación.** Cuentas que copian textos, llegan juntas a las mismas publicaciones en momentos tranquilos, comparten un hashtag propio o se responden en red cerrada. Coincidir en un momento viral casi no pesa; coincidir cuando no hay nadie, mucho. Si una operación se creó en tanda (misma semana, misma forma de usuario), se suman las cuentas de esa tanda.

### Resultados

**Set de práctica, puntaje automático con el script oficial: 54,63 de 55** (narrativas 29,99 de 30; coordinación 24,64 de 25). Sin IA saca 53,87. La línea base del desafío saca 16,95.

| Medida | Valor |
|---|---|
| Agrupamiento de narrativas (B-cubed F1) | 1,000 |
| Detección de ruido (F1) | 1,000 |
| Origen de las 3 narrativas emergentes | 1,000 |
| Postura (macro-F1) | 0,998 (acierta el lado de las 13 narrativas) |
| Cuentas coordinadas (F1) | 0,987: 113 de 115 cuentas de campaña, 1 sola genuina marcada entre 15.876 |
| Operaciones (B-cubed F1) | 0,983 (las 3 campañas, cada una entera en una operación) |

**Set final (150.000 comentarios, 12.002 cuentas, 10 días):**

- 12 narrativas en tres temas (seguridad, festival, dique); 3 nacen con el período ya empezado.
- 20,7% de ruido.
- 3 operaciones coordinadas, 98 cuentas: una empuja que el dique es la solución para el agua (50 cuentas creadas en la misma semana, textos plantilla), otra que el festival es un derroche (26 cuentas antiguas con actividad de cobertura) y otra el cierre de la comisaría del norte (22 cuentas creadas en la misma semana, activas de madrugada).
- **108 segundos desde cero** en una notebook (0,72 s por cada mil comentarios). IA: 6 llamadas por narrativa, US$ 0,19 a precio de lista en total (US$ 0,0012 por cada mil comentarios) y US$ 0 en cada re-corrida, porque las respuestas quedan guardadas.

### Generalización y robustez

Cada cambio se aceptó midiendo en el set de práctica y en un hold-out: un tema entero que el pipeline nunca vio, solo los primeros 10 días, otras semillas y datos ensuciados a propósito. La tabla completa está en `docs/generalizacion.md` y cada corrida queda anotada en `experimentos/runs.csv`, incluidas las ideas que se descartaron (`reportes/mejoras.md`).

### Reproducible

Semillas fijas, versiones exactas en `uv.lock`, validador de formato y 45 tests. Un clon limpio corre con un comando y no necesita ninguna clave: verificamos que sin clave salen los mismos tres archivos, byte a byte.

### Tablero

`reportes/dashboard.html` abre sin internet: línea de tiempo de cada narrativa, apoyo contra rechazo, qué operación la empuja y una ficha por operación (cuentas, antigüedad, horas pico y texto típico).

### Cómo correrlo

```bash
make entrega
```

Baja el set final, instala las versiones exactas y escribe `entrega/` y el tablero. Código abierto bajo licencia MIT.

## Video de demo

`reportes/demo.mp4` (72 segundos, sin audio). Enlace directo una vez pusheado:
https://github.com/valentinodonayo/qualydata-big-data/raw/main/reportes/demo.mp4
