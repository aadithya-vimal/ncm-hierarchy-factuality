import hashlib
import json
from pathlib import Path


def test_repo_scaffold_and_export_contract_exist():
    root = Path(__file__).resolve().parents[1]
    assert (root / "README.md").is_file()
    assert (root / ".gitignore").is_file()
    assert (root / "LICENSE").is_file()
    assert (root / "CITATION.cff").is_file()
    assert (root / "src" / "ncm_factuality" / "dataset_export.py").is_file()
    schema = json.loads((root / "configs" / "dataset_schema.json").read_text())
    assert schema["additionalProperties"] is True


def test_sha256_known_vector():
    assert hashlib.sha256(b"abc").hexdigest() == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
