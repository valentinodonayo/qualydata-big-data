# Atajos del proyecto. Cada "receta" es un comando de una línea.
#
#   make entrega   -> instala todo y corre el pipeline sobre el set FINAL (lo que se entrega)
#   make practica  -> corre sobre el set de práctica y lo puntúa con el script del jurado
#   make test      -> corre los tests
#
# "uv sync --frozen" instala exactamente las versiones de uv.lock (reproducible).

.PHONY: entrega practica datos-final test limpiar

entrega: datos-final
	uv sync --frozen
	uv run python correr.py --datos datos/final --entrega entrega

practica:
	uv sync --frozen
	uv run python evaluacion/evaluar.py --nombre "make-practica"

# Baja el set final del release oficial solo si todavía no está
datos-final:
	@mkdir -p datos/final
	@for f in fuentes.csv publicaciones.csv.gz comentarios.csv.gz cuentas.csv.gz; do \
		[ -f datos/final/$$f ] || curl -sSL -o datos/final/$$f \
		https://github.com/qualydata-org/desafio-qualydata/releases/download/set-final/$$f; \
	done

test:
	uv sync --frozen
	uv run python -m pytest -q tests

# Borra los cálculos guardados (embeddings, UMAP, base DuckDB). La próxima corrida los rehace.
limpiar:
	rm -rf cache
