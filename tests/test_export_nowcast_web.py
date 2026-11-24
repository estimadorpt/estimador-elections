"""The web export's refusals: a run that would put a wrong page online never gets written."""

import json
import os

import pytest

from src.export_nowcast_web import REQUIRED, ExportRefused, check_run, national_rows, polls_since, seat_draws

PARTIES = ["PS", "CH", "AD"]


def _run(tmp_path, **overrides):
    run = {
        "bias_form": "none",
        "last_poll": "2026-11-23",
        "target": "2026-11-23",
        "horizon_days": 0,
        "n_polls_since_last_election": 2,
        "diagnostics": {"passes_gate": True, "divergences": 0, "max_rhat": 1.004},
    }
    run.update(overrides)
    for name in REQUIRED:
        path = tmp_path / name
        if name.endswith(".zarr"):
            path.mkdir()
        else:
            path.write_text("")
    (tmp_path / "run.json").write_text(json.dumps(run))
    return str(tmp_path)


def test_a_passing_run_is_accepted(tmp_path):
    assert check_run(_run(tmp_path))["bias_form"] == "none"


@pytest.mark.parametrize(
    "overrides, reason",
    [
        ({"diagnostics": {"passes_gate": False, "divergences": 0}}, "gate"),
        ({"diagnostics": {"passes_gate": True, "divergences": 3}}, "divergences"),
        ({"bias_form": "static"}, "scenario"),
        ({"target": "2029-10-07", "horizon_days": 1414}, "not a nowcast"),
    ],
)
def test_refuses_runs_that_must_not_be_published(tmp_path, overrides, reason):
    with pytest.raises(ExportRefused, match=reason):
        check_run(_run(tmp_path, **overrides))


def test_refuses_a_run_with_a_file_missing(tmp_path):
    run_dir = _run(tmp_path)
    os.remove(os.path.join(run_dir, "input_polls.csv"))
    with pytest.raises(ExportRefused, match="input_polls.csv"):
        check_run(run_dir)


def test_seat_draws_must_fill_the_parliament(tmp_path):
    (tmp_path / "seat_forecast_simulations.json").write_text(
        json.dumps([{"PS": 100, "CH": 70, "AD": 60, "original_sample_id": 0}, {"PS": 100, "CH": 70, "AD": 59, "original_sample_id": 1}])
    )
    with pytest.raises(ExportRefused, match="229 seats"):
        seat_draws(str(tmp_path), PARTIES)


def test_national_rows_are_fractions_largest_first(tmp_path):
    (tmp_path / "national_summary.csv").write_text(
        "party,median,mean,lo90,hi90,lo50,hi50\nCH,30,30,25,35,28,32\nPS,40,40,35,45,38,42\nAD,30,30,25,35,28,32\n"
    )
    rows = national_rows(str(tmp_path), PARTIES)
    assert [r["party"] for r in rows][0] == "PS"
    assert rows[0]["median"] == pytest.approx(0.4)
    (tmp_path / "national_summary.csv").write_text("party,median,mean,lo90,hi90,lo50,hi50\nPS,40,40,35,45,38,42\nCH,30,30,25,35,28,32\nAD,20,20,15,25,18,22\n")
    with pytest.raises(ExportRefused, match="add up"):
        national_rows(str(tmp_path), PARTIES)


def test_polls_since_keeps_what_the_loader_keeps(tmp_path):
    # A hand entry Wikipedia now lists too, and a row still missing its sample size.
    (tmp_path / "input_polls.csv").write_text(
        "pollster,fieldwork_start,fieldwork_end,sample_size,AD,PS,CH,IL,L,CDU,BE,PAN,JPP,Others,source_url,source\n"
        "Aximage,2026-11-01,2026-11-05,,25,30,26,6,4,3,2,1,,3,,wikipedia\n"
        "Intercampus,2026-11-11,2026-11-18,600,23,28,24,9,4,4,2,3,,3,,hand\n"
        "Intercampus,2026-11-11,2026-11-18,609,22.9,28.2,23.8,8.9,4.1,3.6,2.0,2.6,,3.9,,wikipedia\n"
    )
    polls = polls_since(str(tmp_path), "2025-05-18")
    assert [(p["pollster"], p["sample_size"]) for p in polls] == [("Intercampus", 609)]


def test_polls_since_keeps_only_polls_after_the_election_as_published(tmp_path):
    (tmp_path / "input_polls.csv").write_text(
        "pollster,fieldwork_start,fieldwork_end,sample_size,AD,PS,CH,JPP,Others,source_url\n"
        "Aximage,2025-05-01,2025-05-10,800,32,25,20,,23,\n"
        "Intercampus,2026-11-11,2026-11-18,609,22.9,28.2,23.8,,25.1,https://example.org\n"
    )
    polls = polls_since(str(tmp_path), "2025-05-18")
    assert len(polls) == 1
    assert polls[0]["pollster"] == "Intercampus"
    assert polls[0]["shares"]["PS"] == pytest.approx(0.282)
    assert "JPP" not in polls[0]["shares"]
