update-deps:
	uv sync --upgrade

install:
	uv sync

serve:
	uv run serve

build-image:
	docker build -t yfinance-wrapper .
