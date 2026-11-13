"""Genera un set INVENTADO (temas y textos nuevos) en el mismo formato que el oficial, con su verdad.

Para qué sirve: probar el pipeline con datos que nunca vio (otros temas, otro vocabulario)
y ver qué supuestos se rompen. Nada de acá está copiado de práctica ni del set final.

Uso:
    python evaluacion/generar_set.py --salida datos_inv/gemelo                # mismas reglas que el oficial
    python evaluacion/generar_set.py --salida datos_inv/dif_a --variacion a   # una regla relajada
    python evaluacion/generar_set.py --todos --carpeta datos_inv              # gemelo + las 8 variaciones
    python evaluacion/generar_set.py --todos --carpeta datos_inv --sin-ciclo  # ídem, con gente comentando de madrugada

Variaciones "difíciles" (de a una por set, para saber cuál rompe):
    a  15% de las respuestas largas ACUERDAN con el padre (en el oficial siempre contradicen)
    b  5% de las respuestas cruzan a OTRA narrativa del mismo tema
    c  el ruido a veces está en hilos (10% de los comentarios de ruido)
    d  narrativas del mismo tema con mucho vocabulario compartido
    e  pocos hilos: solo 8% de respuestas
    f  una narrativa muy chica (~300 comentarios) y una muy grande (~30% del set)
    g  apoyo real 35% o 70% (en el oficial es ~50-60%)
    h  campañas distintas: cuentas viejas sin patrón de usuario con textos plantilla + una de solo 8 cuentas
    i  sin ciclo día/noche: la gente común también comenta de madrugada (en el oficial casi nadie)

Todo sale de una semilla fija: el mismo comando genera siempre el mismo set.
"""
import argparse
import gzip
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ZONA = timezone(timedelta(hours=-3))
INICIO = datetime(2026, 10, 5, 0, 0, 0, tzinfo=ZONA)   # el período inventado arranca un lunes
DIAS = 14

# ---------------------------------------------------------------------------------------
# BANCO DE FRASES. Cada narrativa tiene frases que la AFIRMAN ("apoya") y frases que la
# REFUTAN ("rechaza"). El signo "|" separa una cola opcional: a veces se escribe, a veces no.
# ---------------------------------------------------------------------------------------
NARRATIVAS = {
    # ---------------- TEMA: basural a cielo abierto de Colonia Sauce ----------------
    "R1": {"tema": "basural", "titulo": "El humo del basural está enfermando a los vecinos",
           "claves": "humo|quema|asma|respirar|olor",
           "apoya": [
               "el humo del basural nos está enfermando a todos| en el barrio",
               "mis hijos viven con broncoespasmo desde que queman la basura| todas las noches",
               "no se puede respirar cuando prenden fuego el basural| a la tardecita",
               "en la salita hay cola de chicos con asma por la quema| del vaciadero",
               "el olor a plástico quemado se mete en las casas| y no se va más",
               "los abuelos del barrio terminan con oxígeno por culpa del humo",
               "tenemos que dormir con las ventanas cerradas por la humareda| del vertedero",
               "las ratas y las moscas del basural ya están adentro de las casas",
               "el pediatra me dijo que la tos de la nena es por el humo tóxico",
               "queman cubiertas y cables a metros de las casas, nos están envenenando",
               "cada vez hay más vecinos con alergias y problemas de piel| cerca del vaciadero",
               "la ceniza del basural cae arriba de la ropa tendida| y de la huerta",
           ],
           "rechaza": [
               "el humo viene de la quema de pastizales, no del basural",
               "siempre hubo asma en la zona, mucho antes del vaciadero| no inventen",
               "el basural está a cinco kilómetros, el humo no llega hasta el pueblo",
               "los que queman son los mismos vecinos que tiran la basura en cualquier lado",
               "ningún estudio serio demostró que el humo enferme a nadie",
               "en invierno todos los chicos tienen tos, no le echen la culpa a la quema",
               "el viento sopla para el otro lado casi todo el año| el olor ni se siente",
               "exageran con lo del humo para conseguir un terreno del municipio",
               "la salita atiende la misma cantidad de chicos que hace diez años",
               "las ratas aparecen por los galpones de granos, nada que ver con la basura",
               "vivo a dos cuadras del vertedero y jamás me enfermé| son puro verso",
               "el problema son las estufas a leña mal puestas, no el fuego del basural",
           ]},
    "R2": {"tema": "basural", "titulo": "La planta de reciclaje prometida nunca se hizo y la plata no aparece",
           "claves": "planta de reciclaje|prometieron|crédito|galpón vacío|fondos",
           "apoya": [
               "prometieron una planta de reciclaje hace seis años y no pusieron ni un ladrillo",
               "cobraron el crédito para la planta de residuos y la plata no aparece",
               "el galpón de la recicladora está vacío, es un cartel y nada más",
               "anunciaron tres veces la planta de tratamiento y seguimos con el pozo| de siempre",
               "dónde están los fondos que mandó la provincia para sanear el basural",
               "compraron una prensa y una cinta que están oxidándose a la intemperie",
               "la planta de separación fue una promesa de campaña y quedó en la nada",
               "se gastaron el subsidio ambiental en cualquier cosa menos en la recicladora",
               "ni cerco perimetral hicieron y dicen que la planta está en marcha",
               "inauguraron un cartel de obra y nunca más volvió una máquina",
               "el intendente rindió gastos de una planta que no existe",
               "los cartoneros siguen esperando la cooperativa de reciclado que les prometieron",
           ],
           "rechaza": [
               "la planta de reciclaje está en obra, vayan a ver antes de hablar",
               "la plata del crédito recién llegó este año, por eso se demoró| la obra",
               "ya compraron la prensa y la cinta, falta la conexión eléctrica nada más",
               "la obra de la recicladora se frenó por la inflación, no porque falten fondos",
               "la provincia nunca giró la segunda cuota, el municipio no tiene la culpa",
               "el galpón ya está techado y en marzo arranca la separación",
               "la rendición de gastos de la planta está publicada en el boletín municipal",
               "ya hay doce recuperadores trabajando en el predio de la planta",
               "licitaron la obra dos veces y quedó desierta, no es que se robaron la plata",
               "una planta de tratamiento no se hace en un año, tengan paciencia",
               "el cerco y la báscula del predio ya están terminados, yo los vi",
               "los fondos están en una cuenta específica y no se tocaron| pidan el informe",
           ]},
    "R3": {"tema": "basural", "titulo": "Camiones de otra ciudad descargan residuos patológicos de noche",
           "claves": "camiones|de noche|patológicos|jeringas|otra ciudad",
           "apoya": [
               "entran camiones de otra ciudad a descargar de noche en el basural",
               "aparecieron jeringas y bolsas rojas de hospital tiradas en el vaciadero",
               "están trayendo residuos patológicos de la capital| a escondidas",
               "a las tres de la mañana pasan los camiones sin patente rumbo al basural",
               "nos convirtieron en el tacho de basura de toda la región",
               "un chofer contó que le pagan por tirar desechos de clínicas acá",
               "vi tambores con calavera y bidones de químicos entre la basura",
               "los camiones tapados con lona entran por el camino de atrás| cuando no hay nadie",
               "alguien cobra para dejar que descarguen residuos peligrosos de afuera",
               "encontraron sueros, gasas con sangre y agujas donde juegan los pibes",
               "hay un video de los volcadores descargando a la madrugada| circula por los grupos",
               "el basural creció el doble en un mes por la basura que traen de afuera",
           ],
           "rechaza": [
               "los camiones que entran de noche son los recolectores del turno nocturno",
               "las bolsas rojas son de la veterinaria del pueblo, no de ninguna clínica de afuera",
               "el video de los camiones es viejo y de otra provincia",
               "los residuos patológicos del hospital los retira una empresa habilitada",
               "no hay ningún convenio para recibir basura de otros municipios",
               "esos volcadores son de la obra de cloacas, llevan escombro y tierra",
               "revisaron el predio los inspectores y no encontraron nada raro",
               "lo de las jeringas es un audio de whatsapp sin ninguna prueba",
               "el portón del basural tiene candado y sereno después de las diez",
               "ningún chofer va a manejar cien kilómetros para tirar bolsas acá, no cierra",
               "los tambores eran de aceite de un taller, ya los retiraron",
               "inventan lo de los camiones truchos para asustar a la gente",
           ]},
    # ---------------- TEMA: tarifa de la luz de la cooperativa eléctrica ----------------
    "E1": {"tema": "luz", "titulo": "La tarifa de la luz es impagable",
           "claves": "factura|boleta|jubilación|impagable|aumento",
           "apoya": [
               "la factura de luz me vino el triple que el mes pasado",
               "con una jubilación mínima no se puede pagar la boleta de la cooperativa",
               "tuve que elegir entre pagar la luz o comprar los remedios",
               "los comercios están cerrando porque no pueden con la tarifa eléctrica",
               "me llegaron ochenta mil pesos de luz y vivo sola con una heladera",
               "el aumento de la energía es una locura, no hay bolsillo que aguante",
               "la panadería de la esquina apagó un horno porque no le da para la boleta",
               "pagamos la electricidad más cara de toda la provincia",
               "sacaron los subsidios de golpe y la gente no llega a fin de mes",
               "media cuadra tiene la luz cortada por no poder pagar",
               "la tarifa social no la consigue nadie, te piden mil papeles",
               "cada bimestre el recibo viene más alto y los sueldos siguen igual",
           ],
           "rechaza": [
               "durante años pagamos la luz regalada, ahora vale lo que cuesta",
               "si tenés tres aires prendidos todo el día no te quejes de la boleta",
               "la tarifa social existe y se tramita en diez minutos en la oficina",
               "en la capital pagan lo mismo o más, no somos los más caros",
               "la cooperativa compra la energía más cara y no puede regalarla",
               "el que cuida el consumo paga una factura razonable| yo pago poco",
               "sin tarifa actualizada no hay mantenimiento y después se corta",
               "el aumento fue en cuotas y avisaron con meses de anticipación",
               "los comercios cierran por los alquileres, no por la electricidad",
               "hay planes de pago en doce cuotas para el que debe",
               "cambien los focos y la heladera vieja y van a ver cómo baja el recibo",
               "los subsidios eran insostenibles, alguien los tenía que sacar",
           ]},
    "E2": {"tema": "luz", "titulo": "La cooperativa cobra cargos inventados en la factura",
           "claves": "cargos|cuota capital|sepelio|ítems|medidor",
           "apoya": [
               "la mitad de la factura son cargos que nadie sabe qué son",
               "te cobran cuota capital, sepelio y biblioteca adentro de la boleta de luz",
               "los medidores nuevos marcan más de lo que uno gasta",
               "me facturaron un consumo estimado sin venir a leer el medidor",
               "meten el alumbrado público dos veces, en la boleta y en la tasa municipal",
               "la cooperativa inventa ítems para recaudar| y nadie la controla",
               "pedí el detalle de los conceptos y no me lo supieron explicar",
               "el servicio de sepelio es obligatorio aunque no lo quieras, un robo",
               "hay un aporte para obras que se cobra hace diez años y no hicieron ninguna",
               "te cobran reconexión aunque nunca te hayan cortado",
               "el cargo fijo subió más que la energía, ahí está la trampa",
               "nadie audita lo que cobra el consejo de administración en cada recibo",
           ],
           "rechaza": [
               "todos los cargos están aprobados por la asamblea de socios",
               "el sepelio y la biblioteca son servicios que usa medio pueblo| y son optativos",
               "los medidores están homologados y los controla el ente regulador",
               "la cuota capital es lo que te hace dueño de la cooperativa, no un invento",
               "el alumbrado lo fija la ordenanza municipal, la cooperativa solo lo recauda",
               "el detalle de cada concepto está en el dorso de la factura, hay que leerlo",
               "si no querés el servicio de sepelio firmás la baja y listo",
               "el consumo estimado se compensa en la lectura siguiente",
               "el aporte para obras pagó la subestación nueva, no se lo quedó nadie",
               "cualquier socio puede ir a ver el balance y los ítems cuando quiera",
               "el cargo fijo es igual en todas las distribuidoras del país",
               "hicieron contrastar cien medidores y todos daban bien",
           ]},
    "E3": {"tema": "luz", "titulo": "Los cortes de luz son por falta de inversión",
           "claves": "cortes|transformador|baja tensión|se quemó|inversión",
           "apoya": [
               "cada tormenta nos quedamos doce horas sin luz porque no invierten",
               "el transformador del barrio explota todos los veranos y lo emparchan",
               "se me quemó la heladera por la baja tensión y nadie se hace cargo",
               "los cables tienen cuarenta años, por eso se corta a cada rato",
               "pagamos fortunas y nos dan un servicio del siglo pasado",
               "tres apagones en una semana, se nota que no hacen mantenimiento",
               "los postes están podridos y se caen con un viento fuerte",
               "la tensión sube y baja y te arruina los electrodomésticos",
               "la cooperativa no compró un solo transformador nuevo en años",
               "cuando hace calor ya sabemos que a la siesta se va la corriente",
               "tuve que tirar toda la mercadería del freezer por el corte de dos días",
               "la guardia de reclamos no atiende y las cuadrillas no dan abasto",
           ],
           "rechaza": [
               "los cortes son por la línea de alta tensión de la provincia, no por la cooperativa",
               "con las tormentas que hubo se cae cualquier tendido, no es falta de inversión",
               "este año cambiaron dos transformadores y trescientos postes",
               "la gente se cuelga de la red y sobrecarga las líneas, por eso salta",
               "los apagones del verano fueron programados para hacer mejoras",
               "si se te quemó un aparato hacés el reclamo y te lo reconocen",
               "todos enchufan aires a la vez y no hay red que aguante ese pico",
               "la subestación nueva ya está funcionando y bajaron las interrupciones",
               "las cuadrillas trabajaron toda la noche para reponer el servicio",
               "en los pueblos vecinos se corta mucho más seguido que acá",
               "poné un protector de tensión como corresponde y no se te quema nada",
               "el plan de obras está publicado y se viene cumpliendo",
           ]},
    # ---------------- TEMA: la Escuela 214 se cae a pedazos ----------------
    "S1": {"tema": "escuela", "titulo": "La escuela se cae a pedazos y los chicos corren peligro",
           "claves": "techo|se llueve|sin gas|baños|peligro",
           "apoya": [
               "se cayó un pedazo de techo en el aula de tercero| de milagro no había nadie",
               "los chicos estudian con campera porque la escuela no tiene gas",
               "el edificio de la 214 se llueve entero, ponen baldes en los pasillos",
               "hay cables pelados en las aulas, un día va a pasar una desgracia",
               "los baños de la escuela están clausurados hace meses",
               "las paredes están rajadas de punta a punta y nadie viene a mirar",
               "suspenden las clases cada vez que llueve porque se inunda el patio",
               "mi hija volvió con un corte porque se desprendió un vidrio de la ventana",
               "el consejo escolar manda un albañil a pintar y lo de fondo sigue igual",
               "las maestras dan clase en el comedor porque dos salones están apuntalados",
               "hace tres inviernos que prometen las estufas y los pibes se congelan",
               "la instalación eléctrica es de los años setenta, salta la térmica todo el día",
           ],
           "rechaza": [
               "el techo ya lo arreglaron en las vacaciones, no asusten a los padres",
               "el gas se cortó por una pérdida y lo rehabilitan esta semana",
               "la escuela tiene problemas como cualquier edificio viejo, no se cae",
               "vino el arquitecto del ministerio y dijo que la estructura está bien",
               "los baños nuevos se inauguran el mes que viene, ya están los artefactos",
               "son dos aulas con humedad, el resto del edificio está perfecto",
               "la cooperadora compró estufas eléctricas para todos los grados",
               "las clases se suspendieron un solo día en todo el año",
               "las grietas son del revoque, no de las columnas, lo dijo el ingeniero",
               "fui a la reunión de padres y la directora mostró el plan de refacción",
               "cambiaron todo el tablero eléctrico en febrero| ya no salta nada",
               "hay escuelas rurales mucho peores y nadie arma este escándalo",
           ]},
    "S2": {"tema": "escuela", "titulo": "Van a cerrar la escuela y mandar a los chicos a aulas container",
           "claves": "cerrar|fusionar|contenedores|trasladar|matrícula",
           "apoya": [
               "quieren cerrar la 214 y repartir a los chicos en otras escuelas",
               "van a meter a los pibes en contenedores de chapa en vez de arreglar el edificio",
               "ya decidieron fusionar la escuela con la del centro| y no avisaron a nadie",
               "el plan es trasladar toda la matrícula a quince cuadras, los más chicos no pueden ir solos",
               "con el cierre se pierden cargos docentes, por eso no dicen nada",
               "las aulas modulares son hornos en verano y heladeras en invierno",
               "primero la dejan caer y después dicen que conviene cerrarla",
               "circula la resolución que da de baja el establecimiento el año que viene",
               "quieren el terreno de la escuela para otra cosa, por eso la vacían",
               "a los padres de primero ya les dijeron que no anoten a los hermanitos",
               "nos van a dejar sin escuela en el barrio después de sesenta años",
               "los containers son un parche para siempre, lo provisorio queda definitivo",
           ],
           "rechaza": [
               "nadie va a cerrar la escuela, las aulas modulares son mientras dure la obra",
               "el traslado es por seis meses y con transporte gratis para los alumnos",
               "la resolución que circula es falsa, el ministerio ya la desmintió",
               "no se pierde ningún cargo docente, lo firmó el gremio",
               "los módulos tienen aire acondicionado y calefacción, son mejores que las aulas viejas",
               "la inscripción para el año que viene está abierta como siempre",
               "es al revés, van a ampliar la 214 con un salón de usos múltiples",
               "fusionar no es cerrar, comparten el edificio hasta que termine la refacción",
               "el terreno es del estado provincial y no se puede vender",
               "en otras ciudades usaron containers un año y volvieron al edificio arreglado",
               "la supervisora aclaró en la asamblea que la escuela sigue donde está",
               "meten miedo con el cierre para juntar firmas contra la directora",
           ]},
    # ---------------- TEMA: venta del predio del club Juventud Unida ----------------
    "K1": {"tema": "club", "titulo": "Quieren vender el predio del club para hacer un barrio privado",
           "claves": "vender|predio|cancha|inmobiliaria|barrio privado",
           "apoya": [
               "quieren vender la cancha del club para hacer un barrio cerrado",
               "una inmobiliaria ya anduvo midiendo el predio de Juventud Unida",
               "el club es de los socios, no de cuatro vivos que lo quieren lotear",
               "van a rematar las canchitas donde juegan trescientos pibes",
               "llaman a asamblea en enero para aprobar la venta cuando no hay nadie",
               "nos quieren cambiar la pileta y el quincho por torres de departamentos",
               "ese terreno lo donó una familia para el deporte, no para un negocio",
               "si venden la sede el barrio se queda sin el único lugar para los chicos",
               "ya hay un boleto de compraventa firmado y lo esconden",
               "el desarrollador ofreció una fortuna y a varios dirigentes les brillan los ojos",
               "dejaron de cortar el pasto a propósito para decir que el campo de deportes no sirve",
               "le van a poner un paredón y un guardia donde hoy entra cualquiera a jugar",
           ],
           "rechaza": [
               "nadie vende nada, se alquila una parte del predio para pagar deudas",
               "el estatuto prohíbe vender la sede sin el voto de dos tercios de los socios",
               "con esa plata se hace el gimnasio techado que pedimos hace años",
               "la inmobiliaria midió para la escritura, no para lotear",
               "es un lote del fondo que no se usa, las canchas no se tocan",
               "si no entra plata el club quiebra y ahí sí se pierde todo",
               "la asamblea se convocó en la fecha que marca el estatuto",
               "la propuesta es un canje por un predio más grande en la ruta",
               "no hay ningún boleto firmado, pidan el libro de actas",
               "los que protestan hace diez años que no pagan la cuota",
               "el quincho y la pileta siguen siendo de la institución, lo dice el proyecto",
               "prefiero un club chico y ordenado antes que uno grande y fundido",
           ]},
    "K2": {"tema": "club", "titulo": "La comisión directiva se quedó con la plata de los socios",
           "claves": "comisión directiva|balance|cuotas|rifa|tesorero",
           "apoya": [
               "la comisión directiva se quedó con la plata de las cuotas",
               "el balance del club está dibujado, los números no cierran",
               "nadie sabe dónde fue a parar lo recaudado en la rifa de fin de año",
               "el tesorero se compró una camioneta y el club no tiene ni pelotas",
               "hace tres años que no presentan una memoria como corresponde",
               "cobraron el pase de un jugador y esos dólares nunca entraron a la caja",
               "el buffet y la publicidad los manejan los parientes del presidente",
               "pagamos la cuota religiosamente y los vestuarios no tienen agua caliente",
               "pidieron un préstamo a nombre de la institución y nadie vio un peso",
               "los libros contables desaparecieron justo antes de la auditoría",
               "la plata del bingo familiar se evaporó, que rindan cuentas",
               "se votaron sueldos entre ellos en un club que es sin fines de lucro",
           ],
           "rechaza": [
               "el balance lo firmó un contador y lo aprobó la asamblea",
               "la rifa se rindió peso por peso en la cartelera de la sede",
               "la comisión trabaja gratis y encima pone plata de su bolsillo",
               "con las cuotas apenas se paga la luz y el canchero",
               "lo del pase se cobró en cuotas y figura en la memoria",
               "la camioneta del tesorero es de su trabajo, no mezclen",
               "esta directiva agarró el club con juicios y lo sacó adelante",
               "el préstamo fue para el techo del gimnasio y está a la vista",
               "los libros están en la secretaría, cualquier socio los puede consultar",
               "el buffet se concesionó por licitación y ganó la mejor oferta",
               "hablan los de la lista opositora que perdieron las elecciones",
               "denuncien en personas jurídicas si tienen pruebas, si no es puro chusmerío",
           ]},
}

# Cláusulas de contexto por tema: en la variación (d) se pegan a muchos comentarios para que
# las narrativas de un mismo tema compartan MUCHO vocabulario.
CONTEXTO_TEMA = {
    "basural": ["con lo del basural de Colonia Sauce", "por el tema del basural municipal",
                "en el basural de la colonia", "lo del basural a cielo abierto"],
    "luz": ["con la factura de luz de la cooperativa eléctrica", "por el tema de la luz de la cooperativa",
            "lo de la cooperativa eléctrica y la luz", "con la luz de la cooperativa"],
    "escuela": ["con lo de la escuela 214 del barrio", "por el tema del edificio de la escuela 214",
                "lo de la escuela 214", "en la escuela 214 del barrio"],
    "club": ["con lo del club Juventud Unida", "por el tema del club y los socios",
             "lo del club Juventud Unida y la comisión", "en el club del barrio"],
}

TITULARES = {
    "basural": ["Vecinos de Colonia Sauce reclaman por el basural a cielo abierto",
                "El Concejo debate qué hacer con el vaciadero municipal",
                "Nueva quema en el basural: bomberos trabajaron toda la tarde",
                "Ambiente provincial inspeccionó el predio de residuos",
                "Recuperadores urbanos piden mejores condiciones de trabajo"],
    "luz": ["La cooperativa eléctrica anunció el nuevo cuadro tarifario",
            "Usuarios se juntaron frente a la cooperativa por las facturas",
            "Un temporal dejó sin servicio eléctrico a varios barrios",
            "El consejo de administración convocó a asamblea de socios",
            "Comerciantes piden una tarifa diferencial de energía"],
    "escuela": ["Padres de la Escuela 214 reclaman por el estado del edificio",
                "El Consejo Escolar prometió obras para el segundo semestre",
                "Docentes de la 214 hicieron una clase pública en la vereda",
                "El Ministerio evalúa alternativas para el dictado de clases",
                "Ex alumnos organizan una jornada solidaria por la escuela"],
    "club": ["Asamblea caliente en el Club Juventud Unida",
             "Socios de Juventud Unida se movilizan por el futuro del predio",
             "El club presentó su proyecto de infraestructura",
             "Las inferiores de Juventud Unida, entre la ilusión y la incertidumbre",
             "La comisión directiva respondió a las críticas de los socios"],
}

FUENTES = [("f01", "El Diario de la Colonia", "medio"), ("f02", "FM Sauce 98.3", "radio"),
           ("f03", "Canal 5 Regional", "tv"), ("f04", "Portal Sudoeste", "medio"),
           ("f05", "Vecinos Autoconvocados", "pagina"), ("f06", "La Voz del Barrio", "pagina")]

# Ruido: comentarios que no son de ninguna narrativa (saludos, preguntas prácticas, avisos)
RUIDO = [
    "Feliz día para todas las mamás", "Alguien tiene el teléfono del gasista?", "Buen finde gente",
    "Se perdió un perrito marrón por la plaza", "Vendo bici rodado 26, consultar por privado",
    "Mañana atiende el registro civil?", "Qué calor insoportable", "Saludos desde Bahía",
    "Hoy juega la selección?", "Primerooo", "Alguien viaja a la capital el lunes?", "Me avisan cuando empiece",
    "A qué hora abre la farmacia de turno?", "Sigan mi página de tortas", "Qué lindo atardecer hoy",
    "Abrazo grande", "Se cortó internet en todo el centro?", "Pasen la receta", "No entiendo nada",
    "Felicidades a los egresados", "Dónde se saca el carnet de conducir?", "Aguante el campo",
    "Uh qué garrón", "Se viene la lluvia parece", "Etiqueto a mi cuñada", "Hermosa foto",
    "Alguien sabe si hay feria el domingo?", "Excelente nota", "Cuánto sale la inscripción a la colonia de verano?",
    "Fuerza a la familia", "Jajajaja mortal", "Qué grande el Tito", "Hay misa a las 19?",
    "El enlace no me abre", "Paso a saludar", "Dios los bendiga", "Y River cómo salió?",
    "Alquilo departamento de dos ambientes", "Qué sueño tengo", "Me encanta esta radio",
    "Hay colectivo a Colonia Sauce los feriados?", "Suban el video completo", "Quién ganó el sorteo?",
    "Mucha suerte", "Gracias por avisar", "Sin palabras", "Tremendo", "Qué bárbaro", "Vamos todavía",
    "A qué hora es la reunión de la cooperadora?", "Hasta cuándo hay tiempo para pagar la boleta?",
    "El partido del sábado se juega igual?", "Dónde queda el punto verde?", "Miren esto", "Ahí vamos", "😂😂😂", "👏👏", "🙌",
]

CORTAS = ["Tal cual", "Así es", "Exacto", "Totalmente", "Coincido", "Es así", "Clarísimo", "De acuerdo",
          "Lo mismo digo", "Ni más ni menos", "Eso mismo pienso", "Tremendo pero cierto", "Más claro imposible",
          "Sin dudas", "Opino igual"]

PREFIJOS = ["La verdad que", "Lo que nadie dice es que", "Jaja", "Che,", "A ver,", "Y bueno,", "Sinceramente",
            "Te digo algo:", "Ojo,", "Mirá,", "Hay que decirlo:", "Para mí", "Como siempre,", "Escuchen:",
            "Posta,", "Yo les aviso:", "No sé ustedes pero", "Vengo diciendo que"]
SUFIJOS = ["Lamentable.", "así estamos", "Vergüenza.", "Despierten.", "Es así.", "Triste.", "Qué bronca.",
           "Abran los ojos.", "Un desastre.", "No da para más.", "Y nadie dice nada.", "Punto."]
EMOJIS = ["🙄", "😡", "👏", "😔", "🤬", "💯", "🤦", "😤", "👀", "🙏"]
ABREVIATURAS = {"que": "q", "por": "x", "para": "pa", "porque": "xq", "también": "tmb", "de": "d"}
SIN_ACENTOS = str.maketrans("áéíóúÁÉÍÓÚ", "aeiouAEIOU")
NOMBRES = ["marce", "lili", "juanchi", "rober", "caro", "nico", "vale", "tincho", "mabel", "oscar", "pato", "gise",
           "dani", "lucho", "moni", "seba", "anto", "raul", "noe", "kari", "fabi", "gus", "yami", "leo", "sil", "edu"]
APODOS = ["sauce", "sur", "ok", "oficial", "delcampo", "22", "cba", "bahia", "real", "arg", "mdq", "rock"]
PALABRAS_BOT = ["vecino", "alerta", "socio", "verdad", "barrio", "unidos", "patria", "pueblo", "claro", "firme"]


def variar(frase, rng, con_extras=True):
    """Toma una frase base y le aplica variaciones al azar (como hace la gente al escribir)."""
    partes = frase.split("|")
    t = partes[0] + (partes[1] if len(partes) > 1 and rng.random() < 0.5 else "")
    if not con_extras:
        return t[0].upper() + t[1:] + "."
    if rng.random() < 0.45:                                   # prefijo ("La verdad que ...")
        t = rng.choice(PREFIJOS) + " " + t
    else:
        t = t[0].upper() + t[1:]
    if rng.random() < 0.08:                                   # muletilla al final
        t += rng.choice([" eh", ", ojo", ", en serio"])
    t += "."
    if rng.random() < 0.35:                                   # sufijo ("... Lamentable.")
        t += " " + rng.choice(SUFIJOS)
    if rng.random() < 0.12:                                   # abreviaturas: que -> q, por -> x
        t = " ".join(ABREVIATURAS.get(w, w) if rng.random() < 0.7 else w for w in t.split(" "))
    if rng.random() < 0.15:                                   # error de tipeo: dos letras cambiadas de lugar
        palabras = t.split(" ")
        largas = [i for i, w in enumerate(palabras) if len(w) > 4 and w.isalpha()]
        if largas:
            i = int(rng.choice(largas))
            w, j = palabras[i], int(rng.integers(1, len(palabras[i]) - 2))
            palabras[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
            t = " ".join(palabras)
    if rng.random() < 0.05:                                   # letra estirada: "bastaaaa"
        palabras = t.split(" ")
        i = int(rng.integers(0, len(palabras)))
        if palabras[i].isalpha():
            palabras[i] += palabras[i][-1] * int(rng.integers(2, 5))
            t = " ".join(palabras)
    if rng.random() < 0.20:                                   # sin acentos
        t = t.translate(SIN_ACENTOS)
    azar = rng.random()
    if azar < 0.06:                                           # TODO EN MAYÚSCULAS
        t = t.upper()
    elif azar < 0.30:                                         # todo en minúsculas
        t = t.lower()
    if rng.random() < 0.20:
        t += " " + rng.choice(EMOJIS)
    if rng.random() < 0.03:
        t = f"@{rng.choice(NOMBRES)}_{rng.choice(APODOS)} " + t
    return t


def variar_corto(frase, rng):
    """Variaciones suaves para textos cortos (ruido y respuestas tipo 'Tal cual')."""
    t = frase
    if rng.random() < 0.25:
        t = t.translate(SIN_ACENTOS)
    if rng.random() < 0.25:
        t = t.lower()
    if rng.random() < 0.06 and t[-1].isalpha():
        t += t[-1] * int(rng.integers(2, 4))
    if rng.random() < 0.08 and len(t) > 5 and t.replace(" ", "").isalpha():
        j = int(rng.integers(1, len(t) - 2))
        t = t[:j] + t[j + 1] + t[j] + t[j + 2:]
    if rng.random() < 0.12:
        t += " " + rng.choice(EMOJIS)
    return t


# Probabilidad de que un comentario escrito a esa hora "se quede" ahí (medido en el oficial:
# de 2 a 6 de la mañana casi no hay actividad). Si no se queda, pasa a la mañana siguiente.
QUEDA_A_LA_HORA = {0: 0.18, 1: 0.10, 2: 0.04, 3: 0.02, 4: 0.02, 5: 0.02, 6: 0.11, 7: 0.5}


def horario_humano(ts, rng):
    """Corre a la mañana los comentarios que caerían de madrugada (la gente duerme)."""
    hora = int(ts % 86400 // 3600)
    if hora in QUEDA_A_LA_HORA and rng.random() > QUEDA_A_LA_HORA[hora]:
        return ts - ts % 86400 + 7.5 * 3600 + rng.exponential(1.5 * 3600)
    return ts


def iso(segundos):
    """Segundos desde el inicio del período -> fecha ISO con -03:00 (igual que el oficial)."""
    return (INICIO + timedelta(seconds=int(segundos))).isoformat()


def generar(carpeta: Path, variacion: str = "", n_total: int = 60000, semilla: int = 2026, ciclo: bool = True):
    """Genera un set completo y lo escribe en `carpeta` (con su subcarpeta verdad/)."""
    rng = np.random.default_rng(semilla)
    FIN = DIAS * 86400 - 60
    # --- Parámetros (los de "gemelo" imitan lo medido en el set oficial de práctica) ---
    frac_resp = 0.08 if variacion == "e" else 0.14     # respuestas sobre el total (práctica: 14%)
    frac_cortas = 0.20                                 # respuestas cortas ("Tal cual") entre las respuestas
    frac_ruido = 0.19
    p_larga_acuerda = 0.15 if variacion == "a" else 0.0
    p_cruza = 0.05 if variacion == "b" else 0.0
    p_ruido_resp = 0.05 if variacion == "c" else 0.0   # 5% del ruido responde a otro ruido -> ~10% en hilos
    p_contexto = 0.7 if variacion == "d" else 0.0
    # ciclo día/noche (se apaga con ciclo=False o con la variación i)
    humano = (lambda t: t) if (variacion == "i" or not ciclo) else (lambda t: horario_humano(t, rng))

    ids_narr = list(NARRATIVAS)
    temas = sorted({v["tema"] for v in NARRATIVAS.values()})
    # Tamaño relativo de cada narrativa y % de apoyo entre las raíces
    peso = {k: float(rng.uniform(0.6, 1.4)) for k in ids_narr}
    apoyo = {k: float(rng.uniform(0.50, 0.59)) for k in ids_narr}
    emergente = {"R3": 6.3, "S2": 8.8, "K2": 4.09}     # día (con decimales) en que nace cada una
    peso["R3"] *= 0.6; peso["S2"] *= 0.6; peso["K2"] *= 0.35   # las emergentes viven menos días
    cambia_redaccion = {"E1": 8}                       # E1 usa otras frases desde el día 8
    if variacion == "f":
        total_resto = sum(v for k, v in peso.items() if k not in ("E1", "S1"))
        peso["E1"] = total_resto * 0.30 / (0.81 - 0.30) # ~30% del set completo
        peso["S1"] = (total_resto + peso["E1"]) * (300 / n_total) / 0.81
    if variacion == "g":
        for i, k in enumerate(ids_narr):
            apoyo[k] = 0.35 if i % 2 == 0 else 0.70

    # --- Publicaciones: cada una es de un tema, con una hora y una "popularidad" ---
    n_pub = max(40, n_total // 330)
    pub_tema = rng.choice(temas, size=n_pub)
    pub_dia = rng.integers(0, DIAS, size=n_pub)
    pub_hora = rng.choice(np.arange(7, 23), size=n_pub)
    pub_ts = np.sort(pub_dia * 86400 + pub_hora * 3600 + rng.integers(0, 60, size=n_pub) * 60)
    pub_ts[:len(temas)] = 7 * 3600 + np.arange(len(temas)) * 1500    # el primer día hay una nota de cada tema
    pub_tema[:len(temas)] = temas
    pub_peso = rng.lognormal(0, 1.0, size=n_pub)
    pub_id = [f"p{i + 1:05d}" for i in range(n_pub)]
    pub_fuente = rng.choice([f[0] for f in FUENTES], size=n_pub)
    pub_titular = [str(rng.choice(TITULARES[t])) for t in pub_tema]
    # Publicación de origen de cada emergente: la primera de su tema después del día indicado
    origen_ts = {}
    for k, dia in emergente.items():
        del_tema = np.where((pub_tema == NARRATIVAS[k]["tema"]) & (pub_ts >= dia * 86400))[0]
        if k == "K2":                                  # K2 la origina la campaña C3 de madrugada
            origen_ts[k] = dia * 86400
        else:
            origen_ts[k] = float(pub_ts[del_tema[0]]) + 90
            pub_peso[del_tema[0]] = max(pub_peso[del_tema[0]], 4.0)   # la nota que la dispara es popular

    def frase(k, postura, ts):
        """Elige una frase base de la narrativa k con esa postura (respeta el cambio de redacción)."""
        banco = NARRATIVAS[k][postura]
        if k in cambia_redaccion and postura == "apoya":
            mitad = len(banco) // 2
            banco = banco[:mitad] if ts < cambia_redaccion[k] * 86400 else banco[mitad:]
        return str(rng.choice(banco))

    def texto_narrativa(k, postura, ts):
        t = variar(frase(k, postura, ts), rng)
        if p_contexto and rng.random() < p_contexto:   # variación (d): vocabulario compartido del tema
            c = str(rng.choice(CONTEXTO_TEMA[NARRATIVAS[k]["tema"]]))
            t = (c[0].upper() + c[1:] + ", " + t[0].lower() + t[1:]) if rng.random() < 0.6 else (t + " " + c)
        return t

    filas = []   # cada comentario: dict con pub, ts, texto, narrativa, postura, padre (índice) y cuenta

    def agregar(pub, ts, texto, narr, postura, padre=-1, cuenta=None):
        filas.append({"pub": int(pub), "ts": float(min(ts, FIN)), "texto": texto, "narr": narr,
                      "postura": postura, "padre": padre, "cuenta": cuenta})
        return len(filas) - 1

    # --- 1) Raíces: comentarios que no responden a nadie ---
    n_raices = int(n_total * (1 - frac_resp))
    pesos_narr = np.array([peso[k] for k in ids_narr])
    pesos_narr = pesos_narr / pesos_narr.sum() * (1 - frac_ruido)
    eleccion = rng.choice(ids_narr + ["ninguna"], size=n_raices, p=list(pesos_narr) + [frac_ruido])
    pubs_de_tema = {t: np.where(pub_tema == t)[0] for t in temas}
    for k in eleccion:
        if k == "ninguna":
            p = rng.choice(n_pub, p=pub_peso / pub_peso.sum())
            ts = humano(pub_ts[p] + 60 + rng.exponential(5 * 3600))
            agregar(p, ts, variar_corto(str(rng.choice(RUIDO)), rng), "ninguna", "")
            continue
        candidatas = pubs_de_tema[NARRATIVAS[k]["tema"]]
        if k in origen_ts:                              # emergente: solo existe desde su origen
            candidatas = candidatas[pub_ts[candidatas] + 86400 >= origen_ts[k]]
        w = pub_peso[candidatas]
        p = rng.choice(candidatas, p=w / w.sum())
        ts = pub_ts[p] + 60 + rng.exponential(5 * 3600)
        if k in origen_ts and ts < origen_ts[k]:
            ts = origen_ts[k] + rng.exponential(5 * 3600)
        ts = humano(ts)
        postura = "apoya" if rng.random() < apoyo[k] else "rechaza"
        agregar(p, ts, texto_narrativa(k, postura, ts), k, postura)

    # --- 2) Campañas coordinadas ---
    campanas, cuentas_campana = [], {}   # cuentas_campana: nombre de cuenta especial -> (campaña, tipo de cuenta)

    def campana_plantilla(cid, k, n_cuentas, hashtag, plantillas, nuevas=True, de_noche=False, origina=False,
                          por_cuenta=5):
        """Campaña de textos casi idénticos en ráfagas de 12 minutos."""
        cuentas = [f"{cid}_{i}" for i in range(n_cuentas)]
        for c in cuentas:
            cuentas_campana[c] = (cid, "nueva" if nuevas else "vieja")
        candidatas = pubs_de_tema[NARRATIVAS[k]["tema"]]
        n_rafagas = max(4, por_cuenta * 2)
        for r in range(n_rafagas):
            if origina and r == 0:
                inicio = origen_ts[k]
            elif de_noche:
                dia = rng.integers(int(origen_ts.get(k, 0) // 86400) + 1, DIAS)
                inicio = dia * 86400 + rng.uniform(1, 4.5) * 3600
            else:
                dia = rng.integers(0, DIAS)
                inicio = dia * 86400 + rng.uniform(9, 21) * 3600
            previas = candidatas[pub_ts[candidatas] < inicio]
            p = previas[-1] if len(previas) else candidatas[0]
            inicio = max(inicio, pub_ts[p] + 120)
            participan = rng.choice(cuentas, size=max(3, n_cuentas // 2), replace=False)
            ultimo = -1
            for c in participan:
                ts = inicio + rng.uniform(0, 12 * 60)
                if ultimo >= 0 and rng.random() < 0.15:     # a veces se festejan entre ellas con un "Tal cual"
                    agregar(p, filas[ultimo]["ts"] + rng.uniform(5, 200), variar_corto(str(rng.choice(CORTAS)), rng),
                            k, "apoya", padre=ultimo, cuenta=c)
                    continue
                t = str(rng.choice(plantillas))
                if rng.random() < 0.5:
                    t += " " + hashtag
                if rng.random() < 0.15:
                    t += " " + rng.choice(EMOJIS)
                ultimo = agregar(p, ts, t, k, "apoya", cuenta=c)
        campanas.append({"campana_id": cid, "narrativa_id": k, "postura": "apoya", "cuentas": n_cuentas})

    # C1: cuentas nuevas (o viejas en la variación h), plantillas + hashtag propio, ráfagas diurnas
    campana_plantilla("C1", "E2", 60, "#BoletaTrucha",
                      ["Basta de cargos truchos en la boleta de la cooperativa.",
                       "La cooperativa nos roba con ítems inventados en cada factura.",
                       "Que expliquen cada cargo de la boleta, basta de curro.",
                       "Nos cobran cualquier cosa en la factura y nadie controla."],
                      nuevas=(variacion != "h"))
    # C3: cuentas nuevas que ORIGINAN K2 de madrugada con textos casi idénticos
    campana_plantilla("C3", "K2", 25, "#QueRindanCuentas",
                      ["Que la comisión directiva rinda cuentas: dónde está la plata de los socios.",
                       "La comisión se quedó con la plata del club, que rindan cuentas ya.",
                       "Socios: nos están robando la cuota, exijamos el balance."],
                      de_noche=True, origina=True, por_cuenta=3)
    if variacion == "h":
        # C4: campaña mínima de solo 8 cuentas
        campana_plantilla("C4", "R1", 8, "#HumoQueMata",
                          ["El humo del basural nos está matando, cierren el vaciadero ya.",
                           "Nuestros hijos respiran veneno por la quema del basural."], por_cuenta=4)
    # C2: 30 cuentas viejas, textos variados del banco, llegan juntas a las mismas notas y se responden
    c2 = [f"C2_{i}" for i in range(30)]
    for c in c2:
        cuentas_campana[c] = ("C2", "vieja")
    for _ in range(18):
        candidatas = pubs_de_tema["escuela"]
        p = rng.choice(candidatas)
        inicio = pub_ts[p] + rng.uniform(300, 3 * 3600)
        propios = []
        for c in rng.choice(c2, size=12, replace=False):
            ts = inicio + rng.uniform(0, 10 * 60)
            if propios and rng.random() < 0.25:             # se responden entre sí con apoyo corto
                padre = int(rng.choice(propios))
                agregar(p, filas[padre]["ts"] + rng.uniform(5, 300), variar_corto(str(rng.choice(CORTAS)), rng),
                        "S1", "apoya", padre=padre, cuenta=c)
            else:
                propios.append(agregar(p, ts, texto_narrativa("S1", "apoya", ts), "S1", "apoya", cuenta=c))
    campanas.append({"campana_id": "C2", "narrativa_id": "S1", "postura": "apoya", "cuentas": 30})

    # --- 3) Respuestas (hilos) ---
    n_resp = n_total - len(filas)
    con_narr = [i for i, f in enumerate(filas) if f["narr"] != "ninguna"]
    por_tema_activas = lambda tema, ts: [k for k in ids_narr if NARRATIVAS[k]["tema"] == tema
                                         and origen_ts.get(k, 0) <= ts]
    respuestas = []
    for _ in range(n_resp):
        # el padre es casi siempre una raíz; a veces (17%) otra respuesta
        padre = int(rng.choice(respuestas)) if respuestas and rng.random() < 0.17 else int(rng.choice(con_narr))
        f = filas[padre]
        ts = humano(f["ts"] + 20 + rng.exponential(40 * 60))
        k = f["narr"]
        if p_cruza and rng.random() < p_cruza:              # variación (b): responde con OTRA narrativa del tema
            otras = [x for x in por_tema_activas(NARRATIVAS[k]["tema"], ts) if x != k]
            if otras:
                k2 = str(rng.choice(otras))
                postura = "apoya" if rng.random() < apoyo[k2] else "rechaza"
                respuestas.append(agregar(f["pub"], ts, texto_narrativa(k2, postura, ts), k2, postura, padre=padre))
                continue
        if rng.random() < frac_cortas:                      # corta: acompaña al padre
            respuestas.append(agregar(f["pub"], ts, variar_corto(str(rng.choice(CORTAS)), rng), k, f["postura"],
                                      padre=padre))
        else:                                               # larga: contradice al padre (salvo variación a)
            acuerda = p_larga_acuerda and rng.random() < p_larga_acuerda
            postura = f["postura"] if acuerda else ("rechaza" if f["postura"] == "apoya" else "apoya")
            respuestas.append(agregar(f["pub"], ts, texto_narrativa(k, postura, ts), k, postura, padre=padre))
    if p_ruido_resp:                                        # variación (c): ruido que responde a otro ruido
        ruido = [i for i, f in enumerate(filas) if f["narr"] == "ninguna"]
        for hijo in rng.choice(ruido, size=int(len(ruido) * p_ruido_resp), replace=False):
            padre = int(rng.choice(ruido))
            if padre == hijo or filas[padre]["padre"] == hijo or filas[padre]["padre"] >= 0:
                continue
            filas[hijo].update(padre=padre, pub=filas[padre]["pub"],
                               ts=min(filas[padre]["ts"] + 20 + rng.exponential(40 * 60), FIN))

    # --- 4) Cuentas ---
    n_cuentas = max(200, int(n_total / 12.5))
    actividad = rng.lognormal(0, 1.2, size=n_cuentas)
    normales = rng.choice(n_cuentas, size=len(filas), p=actividad / actividad.sum())
    especiales = sorted(cuentas_campana)
    lugar_especial = dict(zip(especiales, rng.choice(n_cuentas, size=len(especiales), replace=False)))
    cuenta_num = np.array([lugar_especial[f["cuenta"]] if f["cuenta"] else normales[i] for i, f in enumerate(filas)])
    # Actividad "de cobertura": las cuentas de C2 (y un poco C1) también comentan otras cosas como cualquiera
    sin_dueno = np.array([i for i, f in enumerate(filas) if not f["cuenta"]])
    for c in especiales:
        extra = 11 if c.startswith("C2") else (1 if c.startswith("C1") else 0)
        if extra:
            cuenta_num[rng.choice(sin_dueno, size=extra, replace=False)] = lugar_especial[c]
    campana_de_num = {lugar_especial[c]: cuentas_campana[c] for c in especiales}
    usuarios, creadas = [], []
    for i in range(n_cuentas):
        camp, tipo = campana_de_num.get(i, ("", ""))
        if tipo == "nueva":                                 # palabra+palabra+4 dígitos, creada días antes
            usuarios.append(f"{rng.choice(PALABRAS_BOT)}{rng.choice(PALABRAS_BOT)}{rng.integers(1000, 9999)}")
            creadas.append((INICIO - timedelta(days=int(rng.integers(1, 9)))).date().isoformat())
        else:
            estilo = rng.integers(0, 4)
            n, a, d = rng.choice(NOMBRES), rng.choice(APODOS), rng.integers(1, 9999)
            usuarios.append([f"{n}_{d}", f"{n}{a}{d % 100}", f"{n}.{d % 100}", f"{n}_{a}{d % 100}"][estilo])
            creadas.append((INICIO - timedelta(days=int(rng.integers(200, 5000)))).date().isoformat())
    seguidores = np.where([campana_de_num.get(i, ("", ""))[1] == "nueva" for i in range(n_cuentas)],
                          rng.integers(0, 40, size=n_cuentas), rng.lognormal(4.5, 1.5, size=n_cuentas).astype(int))
    cuentas = pd.DataFrame({"cuenta_id": [f"u{i + 1:06d}" for i in range(n_cuentas)], "usuario": usuarios,
                            "fecha_creacion": creadas, "seguidores": seguidores,
                            "siguiendo": rng.integers(20, 900, size=n_cuentas)})
    usadas = np.unique(cuenta_num)
    cuentas = cuentas.iloc[usadas].reset_index(drop=True)

    # --- 5) Ordenar por fecha, poner ids y escribir los archivos ---
    orden = np.argsort([f["ts"] for f in filas], kind="stable")
    nuevo_id = {int(viejo): f"c{j + 1:07d}" for j, viejo in enumerate(orden)}
    com = pd.DataFrame({
        "comentario_id": [nuevo_id[int(i)] for i in orden],
        "publicacion_id": [pub_id[filas[i]["pub"]] for i in orden],
        "cuenta_id": [f"u{cuenta_num[i] + 1:06d}" for i in orden],
        "fecha": [iso(filas[i]["ts"]) for i in orden],
        "texto": [filas[i]["texto"] for i in orden],
        "respuesta_a": [nuevo_id[filas[i]["padre"]] if filas[i]["padre"] >= 0 else "" for i in orden],
        "me_gusta": np.minimum(rng.geometric(0.15, size=len(orden)) - 1, 900),
    })
    v_com = pd.DataFrame({"comentario_id": com["comentario_id"],
                          "narrativa_id": [filas[i]["narr"] for i in orden],
                          "postura": [filas[i]["postura"] for i in orden]})
    v_cue = pd.DataFrame({"cuenta_id": cuentas["cuenta_id"],
                          "campana_id": [campana_de_num.get(int(i), ("", ""))[0] for i in usadas]})
    pub = pd.DataFrame({"publicacion_id": pub_id, "fuente_id": pub_fuente, "fecha": [iso(t) for t in pub_ts],
                        "titular": pub_titular})
    pub = pub[pub["publicacion_id"].isin(set(com["publicacion_id"]))]
    # Tabla de narrativas (la fecha de origen es la del PRIMER comentario real de cada una)
    junto = com.merge(v_com, on="comentario_id")
    primero = junto[junto["narrativa_id"] != "ninguna"].groupby("narrativa_id").first()
    impulsada = {c["narrativa_id"]: c["campana_id"] for c in campanas}
    v_narr = pd.DataFrame([{
        "narrativa_id": k, "tema": v["tema"], "titulo": v["titulo"], "frases_clave": v["claves"],
        "emergente": "si" if k in emergente else "no", "fecha_origen": primero.loc[k, "fecha"],
        "publicacion_origen": primero.loc[k, "publicacion_id"],
        "cambia_redaccion_desde_dia": cambia_redaccion.get(k, ""), "impulsada_por": impulsada.get(k, ""),
        "comentarios": int((v_com["narrativa_id"] == k).sum())} for k, v in NARRATIVAS.items()])

    (carpeta / "verdad").mkdir(parents=True, exist_ok=True)
    for df, nombre in ((com, "comentarios.csv.gz"), (cuentas, "cuentas.csv.gz"), (pub, "publicaciones.csv.gz"),
                       (v_com, "verdad/comentarios.csv.gz"), (v_cue, "verdad/cuentas.csv.gz")):
        with gzip.open(carpeta / nombre, "wt", encoding="utf-8") as f:
            df.to_csv(f, index=False)
    v_narr.to_csv(carpeta / "verdad" / "narrativas.csv", index=False)
    pd.DataFrame(campanas).to_csv(carpeta / "verdad" / "campanas.csv", index=False)
    pd.DataFrame(FUENTES, columns=["fuente_id", "nombre", "tipo"]).to_csv(carpeta / "fuentes.csv", index=False)
    resumen = {"escenario": f"inventado-{variacion or 'gemelo'}", "semilla": semilla, "dias": DIAS,
               "publicaciones": len(pub), "comentarios": len(com), "cuentas": len(cuentas),
               "comentarios_ruido": int((v_com["narrativa_id"] == "ninguna").sum()),
               "respuestas": int((com["respuesta_a"] != "").sum()),
               "cuentas_campana": {c["campana_id"]: c["cuentas"] for c in campanas},
               "apoyo_raices": {k: round(apoyo[k], 3) for k in ids_narr},
               "por_narrativa": {k: int(n) for k, n in v_narr.set_index("narrativa_id")["comentarios"].items()}}
    (carpeta / "verdad" / "resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2),
                                                     encoding="utf-8")
    print(f"{carpeta}: {len(com)} comentarios, {len(pub)} publicaciones, {len(cuentas)} cuentas, "
          f"{resumen['respuestas']} respuestas, {resumen['comentarios_ruido']} de ruido")
    return resumen


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--salida", type=Path, help="Carpeta donde escribir UN set")
    ap.add_argument("--variacion", default="", choices=["", "a", "b", "c", "d", "e", "f", "g", "h", "i"],
                    help="Vacío = gemelo (reglas del oficial); una letra = esa regla relajada")
    ap.add_argument("--todos", action="store_true", help="Genera gemelo + las 8 variaciones (a-h)")
    ap.add_argument("--carpeta", type=Path, default=Path("datos_inv"), help="Carpeta base para --todos")
    ap.add_argument("--comentarios", type=int, default=60000)
    ap.add_argument("--semilla", type=int, default=2026)
    ap.add_argument("--sin-ciclo", action="store_true",
                    help="La gente común comenta a cualquier hora (sin esto, de madrugada casi nadie, como en el oficial)")
    args = ap.parse_args()
    if args.todos:
        generar(args.carpeta / "gemelo", "", args.comentarios, args.semilla, not args.sin_ciclo)
        for v in "abcdefgh":
            generar(args.carpeta / f"dif_{v}", v, args.comentarios, args.semilla, not args.sin_ciclo)
    elif args.salida:
        generar(args.salida, args.variacion, args.comentarios, args.semilla, not args.sin_ciclo)
    else:
        ap.error("indicá --salida o --todos")


if __name__ == "__main__":
    main()
