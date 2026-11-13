# Diagnóstico de puntos perdidos (set de práctica)

Entrega analizada: `salida/practica`. Total: **51.98 / 55** (narrativas 29.0, coordinación 22.98).

## 1. Puntos perdidos por sub-métrica

| Sub-métrica | Valor | Peso | Puntos perdidos |
|---|---|---|---|
| Cuentas coordinadas F1 | 0.932 | 17.50 | **1.20** |
| Operaciones B-cubed F1 | 0.891 | 7.50 | **0.82** |
| Postura macro-F1 | 0.916 | 6.00 | **0.50** |
| Agrupamiento B-cubed F1 | 0.981 | 15.00 | **0.29** |
| Ruido F1 | 0.952 | 4.50 | **0.21** |
| Origen de emergentes | 1.000 | 4.50 | **0.00** |
| **Total** | | | **3.02** |

## 2. Narrativas: cómo se reparte cada narrativa real

| Real | Comentarios | Narrativa predicha principal | % en la principal | % a ruido | % a otras |
|---|---|---|---|---|---|
| B1 | 15608 | n05 | 98.6% | 1.4% | 0.0% |
| B2 | 16605 | n02 | 98.6% | 1.4% | -0.0% |
| B3 | 11848 | n08 | 98.5% | 1.5% | 0.0% |
| B4 | 8626 | n11 | 99.0% | 1.0% | -0.0% |
| H1 | 16474 | n03 | 98.8% | 1.2% | -0.0% |
| H2 | 14501 | n06 | 98.9% | 1.1% | 0.0% |
| H3 | 3513 | n13 | 99.8% | 0.2% | 0.0% |
| H4 | 12538 | n07 | 98.2% | 1.8% | 0.0% |
| L1 | 18463 | n01 | 96.3% | 3.7% | 0.0% |
| L2 | 16163 | n04 | 96.7% | 3.3% | 0.0% |
| L3 | 9462 | n10 | 96.4% | 3.6% | 0.0% |
| L4 | 11331 | n09 | 98.6% | 1.4% | 0.0% |
| L5 | 7158 | n12 | 99.2% | 0.8% | 0.0% |

## 3. Ruido

- Ruido real que quedó dentro de una narrativa (falso negativo): **630** (100% son raíces). Ejemplos: «Fui ayer a Villa Tolar, hermoso lugar»; «Fui ayer a la zona, hermoso lugar»; «Fui ayer a la zona, hermoso lugar»; «Fui ayer a la zona, hermoso lugar»
- Comentarios de narrativa marcados como ruido (falso positivo): **3085** (96% raíces; mediana 10 palabras). Ejemplos: «El litio es nuestro!»; «y bueno, los puestos buneos son para gente de afuera, acá so»; «COMO SIEMPRE, QUÉ EMPLEO? TRAEN A TODOS DE AFUERA, A LOS DE »; «El litio es nuestro!»

## 4. Postura por narrativa real

| Real | Acierto | Apoya real | Rechaza real | ¿Bando más grande = apoya? |
|---|---|---|---|---|
| B1 | 99.9% | 7782 | 7606 | sí |
| B2 | 99.8% | 8220 | 8151 | sí |
| B3 | 0.1% | 5745 | 5924 | **no** |
| B4 | 100.0% | 4287 | 4254 | sí |
| H1 | 99.9% | 8252 | 8024 | sí |
| H2 | 99.9% | 7275 | 7063 | sí |
| H3 | 100.0% | 1815 | 1690 | sí |
| H4 | 99.9% | 6174 | 6134 | sí |
| L1 | 100.0% | 9602 | 8182 | sí |
| L2 | 100.0% | 9187 | 6439 | sí |
| L3 | 99.9% | 4906 | 4214 | sí |
| L4 | 98.7% | 6168 | 5010 | sí |
| L5 | 100.0% | 3837 | 3264 | sí |

## 5. Fecha de origen (todas las narrativas, no solo emergentes)

| Real | Emergente | Origen real | Origen estimado | Error (h, + = tarde) |
|---|---|---|---|---|
| L1 | no | 09-01T07:32 | 09-01T07:35 | +0.04 |
| L2 | no | 09-01T07:35 | 09-01T07:39 | +0.06 |
| L3 | no | 09-01T07:33 | 09-01T07:36 | +0.04 |
| L4 | no | 09-01T07:32 | 09-01T07:36 | +0.05 |
| L5 | si | 09-07T08:17 | 09-07T08:18 | +0.01 |
| B1 | no | 09-01T08:32 | 09-01T08:32 | +0.00 |
| B2 | no | 09-01T08:31 | 09-01T08:35 | +0.07 |
| B3 | no | 09-01T08:52 | 09-01T08:52 | +0.00 |
| B4 | si | 09-10T21:32 | 09-10T21:32 | +0.00 |
| H1 | no | 09-01T08:14 | 09-01T08:30 | +0.27 |
| H2 | no | 09-01T08:18 | 09-01T09:32 | +1.23 |
| H3 | si | 09-06T01:12 | 09-06T01:12 | +0.00 |
| H4 | no | 09-01T08:48 | 09-01T08:53 | +0.08 |

Error medio en emergentes: +0.00 h (todas dentro de las 3 h que dan puntaje completo).

## 6. Coordinación

Cruce campaña real × operación detectada:

```
grupo           -  op01  op02  op03
campana_id                         
C1              2    58     0     0
C2              9     0     0    21
C3              2     0    23     0
genuina     15874     1     0     1
```

Falsos negativos por campaña (actividad mediana de las que se escapan):
- C1: 2 cuentas, mediana 3 comentarios en 3 publicaciones
- C2: 9 cuentas, mediana 16 comentarios en 16 publicaciones
- C3: 2 cuentas, mediana 4 comentarios en 2 publicaciones

Falsos positivos: 2 cuenta(s): u001119 (op03, 3 comentarios), u011907 (op01, 277 comentarios)
