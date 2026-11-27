import gc
import sys

import tools


def test_setup_console_log(tmp_path, monkeypatch):
    unraisable = []
    monkeypatch.setattr(sys, "unraisablehook", unraisable.append)
    handle = tools.setup_console_log(tmp_path)
    print("hello")
    handle.close()
    del handle
    gc.collect()
    assert unraisable == []
    assert (tmp_path / "console.log").read_text() == "hello\n"
