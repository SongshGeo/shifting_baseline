# Dependencies are declared in pyproject.toml and locked in uv.lock;
# `uv sync` installs everything (no per-package add step).
setup:
	uv sync

setup-docs:
	uv sync --group docs

docs:
	uv run mkdocs serve

test:
	uv run pytest -vs --clean-alluredir --alluredir tmp/allure_results

report:
	uv run allure serve tmp/allure_results

run:
	uv run python shifting_baseline/abm.py --multirun model=exp model.max_age=30,35,40,45,50,55,60,65 model.memory_baseline=personal,collective model.loss_rate=0.2,0.4,0.6,0.8 model.new_agents=5,10,15,20,25,30,35,40

# --- Sensitivity-analysis result fetching + plotting ------------------------
# Pull Sobol/Morris artifacts from the geany HPC, skipping the heavy
# sample_NNNNNN/ per-sample sub-dirs. Then plot locally with uv.
fetch-sa:
	@command -v rsync >/dev/null 2>&1 || { echo "Error: rsync is not installed"; exit 1; }
	@mkdir -p ./reports/results/sensitivity/
	@echo "Fetching SA results from geany server..."
	@rsync -avzP --partial \
	    --exclude='sample_*/' \
	    --include='*/' \
	    --include='*.csv' \
	    --include='*.json' \
	    --include='*.txt' \
	    --include='*.png' \
	    --exclude='*' \
	    geany:/u/songsh/CodeBase/shifting_baseline/reports/results/sensitivity/ \
	    ./reports/results/sensitivity/ || { \
	        echo "Error: Failed to fetch SA results from geany server"; \
	        echo "Please check:"; \
	        echo "  1. Network connectivity"; \
	        echo "  2. SSH access to geany server (try: ssh geany)"; \
	        echo "  3. Remote path exists: /u/songsh/CodeBase/shifting_baseline/reports/results/sensitivity"; \
	        exit 1; \
	    }
	@echo "SA fetch completed successfully"

plot-sa:
	@uv run python reports/plot_sobol.py

sa: fetch-sa plot-sa
