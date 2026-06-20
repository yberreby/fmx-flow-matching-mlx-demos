# Run every check (default).
check: generate-readme lint typecheck test

# Regenerate README.md from README.md.in with the current module tree.
generate-readme:
    uv run python generate_readme.py

lint:
    uv run ruff check fmx examples

typecheck:
    uv run basedpyright fmx examples

test:
    uv run pytest

# Print the module tree.
overview:
    uv run pypatree
