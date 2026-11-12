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


def _sysctl(clave):
    """Lee un dato del sistema en Mac (en Linux no existe y devuelve None)."""
    try:
        return subprocess.run(["sysctl", "-n", clave], capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return None


def describir_hardware(dispositivo_embeddings=None):
    """Arma un texto tipo 'Apple M5, 10 núcleos, 16 GB RAM; embeddings en GPU integrada (mps)'."""
    partes = [_sysctl("machdep.cpu.brand_string") or platform.processor() or platform.machine(),
              f"{os.cpu_count()} núcleos"]
    memoria = _sysctl("hw.memsize")
    if memoria and memoria.isdigit():
        partes.append(f"{int(memoria) // 2**30} GB RAM")
    texto = ", ".join(partes)
    if dispositivo_embeddings == "mps":
        texto += "; embeddings en la GPU integrada de la Mac (MPS)"
    elif dispositivo_embeddings == "cuda":
        texto += "; embeddings en GPU (CUDA)"
    elif dispositivo_embeddings == "cpu":
        texto += "; todo en CPU, sin GPU"
    return texto


def escribir_costo(carpeta_entrega: Path, crono: Cronometro, comentarios_procesados: int,
                   almacenamiento: str, notas: str, dispositivo_embeddings=None):
    """Escribe entrega/costo.json con el formato que pide el desafío."""
    segundos = crono.total_segundos()
    miles = max(comentarios_procesados, 1) / 1000
    costo = {
        "comentarios_procesados": comentarios_procesados,
        "segundos": segundos,
        "tokens_ia": crono.tokens_ia,
        "usd_ia": round(crono.usd_ia, 6),
        "usd_por_mil_comentarios": round(crono.usd_ia / miles, 6),
        "hardware": describir_hardware(dispositivo_embeddings),
        "almacenamiento": almacenamiento,
        "notas": notas,
        # Extra (no lo pide el formato, pero ayuda al jurado a ver dónde se va el tiempo)
        "segundos_por_etapa": crono.segundos_por_etapa,
    }
    carpeta_entrega.mkdir(parents=True, exist_ok=True)
    (carpeta_entrega / "costo.json").write_text(json.dumps(costo, ensure_ascii=False, indent=2),
                                                encoding="utf-8")
    return costo
