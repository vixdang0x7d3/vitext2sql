# Text-to-SQL

Converts natural language queries into SQL statements for SQLite databases.

## Project structure
(TODO)

## Setup

```bash
git clone https://vixdang0x7d3/vitext2sql
uv sync
```

## Development

Interactive development with marimo notebooks:

```bash
uv run marimo edit notebooks/<file-name>.py
```

## Dependencies

- SQLGlot: SQL parsing
- Marimo: Interactive notebooks
- Pandas: Data handling
- Transformers, OpenAI, LlamaCpp: LLM providers
