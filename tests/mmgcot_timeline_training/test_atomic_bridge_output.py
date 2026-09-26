import json
from concurrent.futures import ThreadPoolExecutor

from mmgcot_timeline_training.generate_bridge_v2 import _write_exclusive


def test_exclusive_protocol_write_is_complete_under_contention(tmp_path) -> None:
    path = tmp_path / "protocol.json"

    def write(index: int):
        try:
            _write_exclusive(path, {"index": index, "payload": "x" * 10000})
            return "created"
        except FileExistsError:
            # The competing writer must see the complete JSON document.
            assert len(json.loads(path.read_text())["payload"]) == 10000
            return "exists"

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(write, range(8)))
    assert results.count("created") == 1
    assert results.count("exists") == 7
    assert not list(tmp_path.glob(".protocol.json.*"))
