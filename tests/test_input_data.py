import pickle
from pathlib import Path

from ncm_factuality.input_data import decode_reference, load_pickles


class FakeTokenizer:
    eos_token_id = 9
    pad_token_id = 0

    def decode(self, ids, skip_special_tokens=True):
        return ",".join(str(i) for i in ids)


def test_decode_reference_stops_at_eos_and_ignores_padding():
    assert decode_reference([1, 0, 2, 9, 3, 0], FakeTokenizer()) == "1,2"
    assert decode_reference("  reference text  ", FakeTokenizer()) == "reference text"


def test_load_official_format_pickles_respects_explicit_configuration(tmp_path: Path):
    with (tmp_path / "passages.pickle").open("wb") as handle:
        pickle.dump([[1, 2], [3, 4]], handle)
    with (tmp_path / "references.pickle").open("wb") as handle:
        pickle.dump(["one", "two"], handle)
    cfg = {"data": {"ncm_input_file": "passages.pickle", "ncm_reference_file": "references.pickle", "max_examples": None}}
    passages, references = load_pickles(tmp_path, cfg)
    assert passages == [[1, 2], [3, 4]]
    assert references == ["one", "two"]
    cfg["data"]["max_examples"] = 1
    passages, references = load_pickles(tmp_path, cfg)
    assert passages == [[1, 2]]
    assert references == ["one"]
