.PHONY: install install-dev test backtest run docker-build docker-run

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

test:
	pytest tests/ -v

backtest:
	python scripts/run_backtest.py

run:
	streamlit run app.py

docker-build:
	docker build -t supply-chain-cockpit .

docker-run:
	docker compose up
