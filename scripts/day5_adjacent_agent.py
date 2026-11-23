"""Day 5 experiment: vector search with adjacent-page expansion."""

import argparse
import json
import sqlite3
from time import perf_counter

from google.genai.errors import APIError
from pydantic_ai import Agent

from app.agent import INSTRUCTIONS, build_model, print_trace
from app.config import MODEL, api_error_message
from app.retrieval import search_with_neighbors


def search_materials(query: str) -> str:
    """Search the main materials for Lectures 01–03 and return relevant page text, titles, and page numbers.

    Use query to specify what you want to look up.
    """
    try:
        results = search_with_neighbors(query, radius=1)
        return json.dumps({"results": results}, ensure_ascii=False)
    except (ValueError, FileNotFoundError, sqlite3.Error) as exc:
        return json.dumps(
            {"error": str(exc), "results": []},
            ensure_ascii=False,
        )
    except APIError as exc:
        return json.dumps(
            {"error": api_error_message(exc.code), "results": []},
            ensure_ascii=False,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()

    agent = Agent(instructions=INSTRUCTIONS)
    agent.tool_plain(search_materials)

    print(f"[model] {MODEL}")
    print("[retrieval] Vector search + adjacent pages (radius=1)")

    started = perf_counter()
    result = agent.run_sync(args.question, model=build_model())
    elapsed = perf_counter() - started

    print(result.output)

    if args.trace:
        print_trace(result, elapsed)


if __name__ == "__main__":
    main()
