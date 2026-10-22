import json
from pathlib import Path


def test_dataset_schema_is_valid_json_and_documents_public_tables():
    path = Path(__file__).resolve().parents[1] / "configs" / "dataset_schema.json"
    schema = json.loads(path.read_text(encoding="utf-8"))
    assert "NCM Hierarchy Factuality" in schema["title"]
    assert "supported" in schema["properties"]["label"]["enum"]
    assert "unresolved" in schema["properties"]["label"]["enum"]
