"""Compatibilidade para o harness E2, que importa o módulo main."""
from pipeline import create_llm, query_api

__all__ = ["create_llm", "query_api"]
