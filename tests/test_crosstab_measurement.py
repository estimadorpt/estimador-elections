import csv
import json

import pandas as pd

from src.evaluation import crosstab_measurement as cm


def _deposit(path, num, labels, rows):
    matrix = {
        "heading": "Intenção de voto",
        "base_description": "Base: total",
        "column_labels": labels,
        "rows": [{"party_raw": p, "party_normalized": p, "values": v} for p, v in rows],
    }
    (path / f"{num}.json").write_text(json.dumps({"deposit_number": num, "matrices": [matrix]}))


def _setup(tmp_path):
    deps = tmp_path / "deps"
    deps.mkdir()
    # a tracker table whose first label has no value: every value sits one label to the left
    _deposit(
        deps,
        "2025047",
        ["Voto S/Distribuição", "Global", "Norte", "Centro", "Lisboa", "Sul + Ilhas"],
        [
            ("PS", [22, 20, 26, 20, 24, None]),
            ("AD", [28, 30, 28, 24, 29, None]),
            ("CHEGA", [13, 11, 13, 12, 18, None]),
            ("IL", [5, 6, 4, 5, 4, None]),
            ("Indecisos", [17, 17, 16, 18, 16, None]),
        ],
    )
    # a table whose Centro column repeats the total: shifted values, not usable
    _deposit(
        deps,
        "2025025",
        ["Total", "Norte", "Centro", "Lisboa", "Alentejo", "Algarve"],
        [
            ("PS", [21, 20, 21, 23, 25, 19]),
            ("AD", [30, 33, 30, 27, 26, 28]),
            ("CHEGA", [15, 14, 15, 13, 20, 22]),
            ("IL", [6, 6, 6, 8, 3, 4]),
        ],
    )
    with open(tmp_path / "deposits.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["deposit_number", "company", "deposit_datetime"])
        w.writeheader()
        w.writerow(
            {"deposit_number": "2025047", "company": "Pitagórica", "deposit_datetime": "2025-05-10T10:00"}
        )
        w.writerow(
            {"deposit_number": "2025025", "company": "Consulmark2", "deposit_datetime": "2025-04-03T10:00"}
        )
    pd.DataFrame(
        {
            "municipio_code": ["1312", "1106", "0201", "0805", "1001", "4301"],
            "nuts3_code": ["11A", "170", "184", "150", "16B", "200"],
            "nuts2_code": ["11", "17", "18", "15", "16", "20"],
        }
    ).to_parquet(tmp_path / "nuts.parquet")
    cm.setup(str(deps), str(tmp_path / "deposits.csv"), str(tmp_path / "nuts.parquet"))


def test_load_realigns_stub_label_and_drops_shifted_tables(tmp_path):
    _setup(tmp_path)
    tables, notes = cm.load()
    assert [t["deposit"] for t in tables] == ["2025047"]
    t = tables[0]
    assert t["total_col"] == 0
    regions = {label: j for j, label, _ in t["regions"]}
    assert list(regions) == ["Norte", "Centro", "Lisboa", "Sul + Ilhas"]
    assert t["values"]["PS"][0] == 22 and t["values"]["PS"][regions["Norte"]] == 20
    assert t["values"]["CH"][regions["Sul + Ilhas"]] == 18
    # the share naming a party: 22+28+13+5 of the 85 the total column lists (undecided included)
    assert abs(t["named"] - 68 / 85) < 1e-9
    assert any(num == "2025025" and "repeat the total" in why for num, why in notes)


def test_old_metropolitan_regions_for_2011_style_schemes(tmp_path):
    _setup(tmp_path)
    assert cm.munis("Gr. Porto", "nonstandard") == cm.OLD_GR_PORTO
    assert cm.munis("GRANDE LISBOA", "nonstandard") == cm.OLD_GR_LISBOA
    assert cm.munis("Litor. Nort", "nonstandard") is None
    assert cm.munis("Sul + Ilhas", "nuts2") == {"0201", "0805", "4301"}
