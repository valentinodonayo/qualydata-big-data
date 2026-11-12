"""Pipeline de QualyData: narrativas y coordinación a escala.

Cada archivo de esta carpeta es una etapa del pipeline:
- datos.py:  leer, limpiar y guardar los datos (DuckDB + Parquet)
- costo.py:  cronometrar etapas y armar costo.json
- salida.py: escribir los 4 archivos de entrega/ con el formato del jurado
"""
