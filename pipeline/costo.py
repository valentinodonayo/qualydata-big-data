"""Mide cuánto tarda y cuánto cuesta cada corrida, y arma costo.json.

El jurado puntúa "costo, escala y almacenamiento" mirando costo.json.
Por eso no lo escribimos a mano: lo llena el propio pipeline mientras corre.
"""
import json
import os
import platform
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path


class Cronometro:
    """Junta los segundos de cada etapa y lo gastado en IA."""

    def __init__(self):
        self.inicio = time.perf_counter()
        self.segundos_por_etapa = {}   # ej: {"datos": 3.2, "narrativas": 120.5}
        self.tokens_ia = 0             # tokens de entrada + salida de todas las llamadas a la IA
        self.usd_ia = 0.0              # dólares gastados en IA

    @contextmanager
    def etapa(self, nombre):
        """Se usa así:  with crono.etapa("datos"): ...  y mide cuánto tardó ese bloque."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            # "finally" corre aunque la etapa falle: así el tiempo queda anotado igual
            self.segundos_por_etapa[nombre] = round(time.perf_counter() - t0, 2)

    def sumar_ia(self, tokens, usd):
        """Lo llama el módulo de IA después de cada llamada."""
        self.tokens_ia += int(tokens)
        self.usd_ia += float(usd)

    def total_segundos(self):
        return round(time.perf_counter() - self.inicio, 2)


def describir_hardware():
    """Arma un texto tipo 'arm64, 10 núcleos, 16 GB RAM, sin GPU dedicada'."""
    partes = [platform.machine(), f"{os.cpu_count()} núcleos"]
    try:
        # En Mac, sysctl devuelve la memoria total en bytes
        bytes_ram = int(subprocess.run(["sysctl", "-n", "hw.memsize"],
                                       capture_output=True, text=True, timeout=5).stdout)
        partes.append(f"{bytes_ram // 2**30} GB RAM")
    except Exception:
        pass  # si no se puede averiguar (ej: Linux), lo omitimos
    partes.append("sin GPU dedicada")
    return ", ".join(partes)


def escribir_costo(carpeta_entrega: Path, crono: Cronometro, comentarios_procesados: int,
                   almacenamiento: str, notas: str):
    """Escribe entrega/costo.json con el formato que pide el desafío."""
    segundos = crono.total_segundos()
    miles = max(comentarios_procesados, 1) / 1000
    costo = {
        "comentarios_procesados": comentarios_procesados,
        "segundos": segundos,
        "tokens_ia": crono.tokens_ia,
        "usd_ia": round(crono.usd_ia, 6),
        "usd_por_mil_comentarios": round(crono.usd_ia / miles, 6),
        "hardware": describir_hardware(),
        "almacenamiento": almacenamiento,
        "notas": notas,
        # Extra (no lo pide el formato, pero ayuda al jurado a ver dónde se va el tiempo)
        "segundos_por_etapa": crono.segundos_por_etapa,
    }
    carpeta_entrega.mkdir(parents=True, exist_ok=True)
    (carpeta_entrega / "costo.json").write_text(json.dumps(costo, ensure_ascii=False, indent=2),
                                                encoding="utf-8")
    return costo
