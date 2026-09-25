"""The stamp a run writes is the stamp a loader checks -- and drift is refused.

No Isaac needed: ``contract.verify`` is stdlib only.
"""
from __future__ import annotations

import json

import pytest

from contract import verify


def test_stamp_round_trips_through_a_run_dir(tmp_path):
    path = verify.write_stamp(tmp_path, "v2")
    assert path == tmp_path / verify.STAMP_FILENAME
    stamped = json.loads(path.read_text())
    assert stamped["contract_version"] == "v2"
    assert stamped["contract_hash"] == verify.contract_hash("v2")
    assert verify.read_stamp(tmp_path) == stamped
    assert verify.check_run_dir(tmp_path) is True


def test_a_run_without_a_stamp_is_reported_not_refused(tmp_path):
    assert verify.read_stamp(tmp_path) is None
    assert verify.check_run_dir(tmp_path) is False


def test_a_stamp_from_a_different_contract_is_refused(tmp_path):
    verify.write_stamp(tmp_path, "v2")
    stamped = verify.read_stamp(tmp_path)
    stamped["contract_hash"] = "0" * 64
    (tmp_path / verify.STAMP_FILENAME).write_text(json.dumps(stamped))
    with pytest.raises(verify.ContractMismatch):
        verify.check_run_dir(tmp_path)


def test_v1_and_v2_stamps_differ(tmp_path):
    a = verify.write_stamp(tmp_path / "a", "v1").read_text()
    b = verify.write_stamp(tmp_path / "b", "v2").read_text()
    assert json.loads(a)["contract_hash"] != json.loads(b)["contract_hash"]


def test_version_follows_obs_version_env(monkeypatch):
    monkeypatch.delenv("OBS_VERSION", raising=False)
    assert verify.version_from_env() == "v2"
    monkeypatch.setenv("OBS_VERSION", "V1")
    assert verify.version_from_env() == "v1"


def test_readme_quotes_the_real_v2_hash():
    from pathlib import Path

    # aigp-sim's README moved to docs/campaign/sim/README.md in the consolidated repo.
    readme = (
        Path(__file__).resolve().parents[2] / "docs" / "campaign" / "sim" / "README.md"
    ).read_text()
    assert verify.short_hash("v2") in readme
