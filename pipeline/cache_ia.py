"""Caché versionada de respuestas de la IA, para que el jurado reproduzca la entrega SIN clave.

Cómo funciona:
- Cada pedido a la IA se identifica con una "huella" (hash SHA-256) de: modelo + parámetros +
  instrucciones + mensaje. Si cambia cualquier cosa del pedido, cambia la huella.
- La respuesta se guarda en modelos/cache_ia.json, que SÍ va al repositorio. Solo guarda la
  respuesta del modelo y cuántos tokens/dólares costó: nunca claves ni datos personales.
- Al correr: si la huella está en la caché, se usa esa respuesta (no hace falta clave ni internet).
  Si no está y hay clave, se llama al modelo y se agrega a la caché. Si no está y no hay clave,
  el pipeline sigue con la regla sin IA y lo avisa en el log.

Así, con la caché del repo, cualquiera que corra el comando obtiene exactamente la misma entrega.
"""
import hashlib
import json
import threading
from pathlib import Path

RUTA = Path(__file__).resolve().parent.parent / "modelos" / "cache_ia.json"
_candado = threading.Lock()   # la IA se llama en paralelo: el candado evita escrituras pisadas


def huella(modelo: str, parametros: dict, sistema: str, mensaje: str) -> str:
    """Hash estable del pedido completo (json con claves ordenadas: no depende del orden)."""
    pedido = json.dumps({"modelo": modelo, "parametros": parametros, "sistema": sistema,
                         "mensaje": mensaje}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(pedido.encode("utf-8")).hexdigest()


class CacheIA:
    def __init__(self, ruta: Path = RUTA):
        self.ruta = Path(ruta)
        self.datos = {}
        if self.ruta.exists():
            try:
                self.datos = json.loads(self.ruta.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self.datos = {}      # caché rota: se ignora (se regenera si hay clave)
        self.aciertos = 0
        self.nuevas = 0

    def buscar(self, clave: str):
        entrada = self.datos.get(clave)
        if entrada is not None:
            self.aciertos += 1
        return entrada

    def guardar(self, clave: str, entrada: dict):
        """entrada = {"respuesta": {...}, "tokens_entrada": n, "tokens_salida": n, "usd": x}"""
        with _candado:
            self.datos[clave] = entrada
            self.nuevas += 1

    def escribir(self):
        """Escribe el archivo solo si hubo respuestas nuevas (ordenado, para diffs limpios en git)."""
        if not self.nuevas:
            return
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        self.ruta.write_text(json.dumps(self.datos, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                             encoding="utf-8")
