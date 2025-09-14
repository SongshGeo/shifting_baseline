setup:
	make install-tests
	make install-jupyter
	make setup-organizor

setup-organizor:
	poetry add hydra-core

install-jupyter:
	poetry add ipykernel --group dev

install-tests:
	poetry add pytest allure-pytest --group dev
	poetry add pytest-clarity pytest-sugar --group dev

# https://timvink.github.io/mkdocs-git-authors-plugin/index.html
install-docs:
	poetry add --group docs mkdocs mkdocs-material
	poetry add --group docs mkdocs-git-revision-date-localized-plugin
	poetry add --group docs mkdocs-minify-plugin
	poetry add --group docs mkdocs-redirects
	poetry add --group docs mkdocs-awesome-pages-plugin
	poetry add --group docs mkdocs-git-authors-plugin
	poetry add --group docs mkdocstrings\[python\]
	poetry add --group docs mkdocs-bibtex
	poetry add --group docs mkdocs-macros-plugin
	poetry add --group docs mkdocs-jupyter
	poetry add --group docs mkdocs-callouts
	poetry add --group docs mkdocs-glightbox

test:
	poetry run pytest -vs --clean-alluredir --alluredir tmp/allure_results

report:
	poetry run allure serve tmp/allure_results

run:
	poetry run python shifting_baseline/abm.py --multirun model=exp model.max_age=30,35,40,45,50,55,60,65 model.memory_baseline=personal,collective model.loss_rate=0.2,0.4,0.6,0.8 model.new_agents=5,10,15,20,25,30,35,40
