
.PHONY: install test lint format run

install:
	uv sync --all-groups --link-mode copy

test:
	uv run python -m pytest -v

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

run:
	uv run python -c "from streaming_chat import __version__; print(f'Streaming Chat CLI v{__version__} initialized. Chat functionality arrives in Step 8.')"
