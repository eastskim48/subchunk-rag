"""Small JSON query parsing helpers."""

import json


def parse_json_query(json_query: str):
    parsed = json.loads(json_query)
    return parsed["query"]


def parse_json_query_record(json_query: str):
    parsed = json.loads(json_query)
    if not isinstance(parsed, dict):
        raise ValueError("query record must be a JSON object")
    if "id" not in parsed:
        raise ValueError("query record is missing required id")
    if "query" not in parsed:
        raise ValueError("query record is missing required query")
    return {"id": parsed["id"], "query": parsed["query"]}
