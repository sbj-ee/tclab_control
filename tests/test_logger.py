"""Tests for RunLogger (no hardware required)."""

import csv

from tclab_control.logger import RunLogger


def test_logger_writes_header_and_rows(tmp_path):
    path = tmp_path / "run.csv"
    with RunLogger(path) as log:
        log.log(t=0.0, T1=22.0, T2=23.0, Q1=0.0, Q2=0.0)
        log.log(t=1.0, T1=22.5, T2=23.1, Q1=60.0, Q2=0.0)

    rows = list(csv.DictReader(path.open()))
    assert [r["t"] for r in rows] == ["0.0", "1.0"]
    assert rows[1]["Q1"] == "60.0"
    assert rows[0]["T2"] == "23.0"
