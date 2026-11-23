"""Day 5 Advanced: selective graph retrieval agent."""

import argparse
import json
import sqlite3
from time import perf_counter

from google.genai.errors import APIError
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError

from app.advanced_retrieval import search_graph_selective
from app.agent import GRAPH_INSTRUCTIONS, build_model, print_trace
from app.config import MODEL, api_error_message


def search_graph(query: str) -> str:
    """Search the materials, then follow adjacent pages and shared terms to return related page text and citation metadata.

    Use query to specify what you want to look up.
    """
    try:
        return json.dumps(
            search_graph_selective(query),
            ensure_ascii=False,
        )
    except (ValueError, OSError, sqlite3.Error) as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False)
    except APIError as exc:
        return json.dumps(
            {"error": api_error_message(exc.code)},
            ensure_ascii=False,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()

    agent = Agent(instructions=GRAPH_INSTRUCTIONS)
    agent.tool_plain(search_graph)

    print(f"[model] {MODEL}")
    print("[retrieval] Selective Graph: 4 pages / 5000 content chars")

    started = perf_counter()
    try:
        result = agent.run_sync(args.question, model=build_model())
    except ModelHTTPError as exc:
        print(api_error_message(exc.status_code, str(exc.body)[:300]))
        raise SystemExit(1) from None
    elapsed = perf_counter() - started

    print(result.output)

    if args.trace:
        print_trace(result, elapsed)


if __name__ == "__main__":
    main()
