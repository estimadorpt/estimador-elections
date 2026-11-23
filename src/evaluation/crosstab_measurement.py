"""Regional crosstabs in the ERC deposits: a measurement study before any model (protocol section 6).

Reads the v2 parser's output for every deposit in the final 8 weeks before an election, keeps the
regional vote-intention table, maps each pollster's regions to municipalities, and measures:

* the pooled sample: distinct interviews that name a party (a rolling tracker counts only its new
  interviews, sample sizes from data/marktest_polls.csv);
* the design effect: scatter between a pollster's non-overlapping tables against the binomial;
* the bias: how far a pollster's regional deviation from its own national figure, log(s_rp / s_Tp),
  lies from the true one, log(true_rp / true_Up), rms over regions and parties with 5% or more,
  minus the sampling variance; pooled over pollsters on the regions they share; and whether a
  pollster's errors repeat from one election to the next.

Stop rule agreed on 23 November 2026: stop if the effective sample is under 10,000 or the bias is above
0.07 on the log scale. The deposits and the NUTS file are not in this repository:

    python -m src.evaluation.crosstab_measurement \
        --deposits ~/code/estimador-data/data/polls_v2_output \
        --deposits-csv ~/code/estimador-data/data/polls/erc/deposits.csv \
        --nuts ~/code/estimador-microsynthesis/data/raw/ine/municipio_nuts3_2013.parquet

Region definitions: Marktest and Intercampus (2011-2015) use the old metropolitan NUTS-III areas
(Grande Lisboa 20%, Grande Porto 11-13% of their samples, from the fichas técnicas); Intercampus,
Metris and Consulmark NUTS-II 2013 (Consulmark's last 2025 wave NUTS-II 2024); Aximage, Eurosondagem
and Duplimétrica split the Porto metropolitan area from Norte. Regions that cannot be mapped
(Litoral, Interior) are dropped and counted.
"""

from __future__ import annotations

import argparse, collections, csv, datetime as dt, glob, json, os, re

import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data")
PATHS = {}
DEP: dict = {}
N2: dict = {}
N3: dict = {}
ISLANDS: set = set()
MAINLAND: set = set()
polls = None


def setup(deposits, deposits_csv, nuts):
    global DEP, N2, N3, ISLANDS, MAINLAND, polls
    PATHS["deposits"] = deposits
    DEP = {r["deposit_number"]: r for r in csv.DictReader(open(deposits_csv))}
    n = pd.read_parquet(nuts)
    n["dico"] = n.municipio_code.astype(str).str.zfill(4)
    N3.update(zip(n.dico, n.nuts3_code.astype(str)))
    N2.update(zip(n.dico, n.nuts2_code.astype(str)))
    ISLANDS |= {k for k, v in N2.items() if v in ("20", "30")}
    MAINLAND |= set(N2) - ISLANDS
    polls = pd.read_csv(os.path.join(DATA, "marktest_polls.csv"))
    for c in ("Fieldwork Start", "Fieldwork End"):
        polls[c] = pd.to_datetime(polls[c], dayfirst=True, errors="coerce").dt.date
    polls["Pollster"] = polls.Pollster.str.strip(" *")


REG = re.compile(
    r"norte|centro|lisboa|porto|sul|alentejo|algarve|interior|litoral|^a\.?m\.?[lp]|^am[lp]|ilhas|a[çc]ores|madeira|gr\.? ",
    re.I,
)
AD = re.compile(
    r"^(PSD|AD|PPD/PSD|PSD/CDS|PSD\.CDS|PSD/CDS-PP|PSD.CDS-PP.PPM|PORTUGAL À FRENTE|PAF|PÀF|ALIANÇA DEMOCRÁTICA|NOVA AD)",
    re.I,
)
EXCL = re.compile(
    r"governo|coliga[çc][ãa]o.*(prefer|deveria|cen[áa]rio)|l[íi]der|din[âa]mica|vit[óo]ria|ganhar|maioria|simpatia|proximidade|identifica|cen[áa]rio|segunda volta|\bPR\d|presiden|confian[çc]a|avalia|actua|atua|melhor|primeiro|resultado.*anterior|votou|voto anterior|recorda",
    re.I,
)


def label(m):
    return (m.get("heading") or "") + " | " + (m.get("base_description") or "")


def candidates(d):
    out = []
    for i, m in enumerate(d.get("matrices") or []):
        cl = [str(c) for c in (m.get("column_labels") or [])]
        rows = [
            str(x.get("party_normalized") or x.get("party_raw") or "").strip() for x in (m.get("rows") or [])
        ]
        if sum(bool(REG.search(c)) for c in cl) < 3:
            continue
        if "PS" not in rows or not any(AD.match(r) for r in rows):
            continue
        if len(rows) < 4 or EXCL.search(label(m)):
            continue
        out.append(i)
    return out


PARTIES = ["PS", "AD", "CH", "IL", "BE", "CDU", "L", "PAN"]
OLD_GR_PORTO = {
    "0107",
    "1304",
    "1306",
    "1308",
    "1312",
    "1313",
    "1315",
    "1316",
    "1317",
}  # NUTS-III Grande Porto (2002)
OLD_GR_LISBOA = {
    "1115",
    "1105",
    "1106",
    "1107",
    "1109",
    "1116",
    "1110",
    "1111",
    "1114",
}  # NUTS-III Grande Lisboa (2002)
SETUBAL_SOUTH_AML = {
    "1503",
    "1504",
    "1505",
    "1506",
    "1507",
    "1508",
    "1510",
    "1511",
    "1512",
}  # Alcochete..Setúbal (AML part of Setúbal district)


def party_votes(year):
    d = pd.read_parquet(f"{DATA}/legislativas_{year}.parquet")
    code = "territoryCode" if "territoryCode" in d else "territory_code"
    d["dico"] = d[code].str.replace("LOCAL-", "").str[:4]
    cols = {
        "PS": [c for c in d if c == "PS"],
        "AD": [c for c in d if c.startswith("PPD/PSD") or c in ("CDS-PP", "CDS-PP.PPM")],
        "CH": [c for c in d if c == "CH"],
        "IL": [c for c in d if c == "IL"],
        "BE": [c for c in d if c == "B.E."],
        "CDU": [c for c in d if c == "PCP-PEV"],
        "L": [c for c in d if c in ("L", "L/TDA")],
        "PAN": [c for c in d if c == "PAN"],
    }
    out = pd.DataFrame({p: d[c].fillna(0).sum(1) if c else 0.0 for p, c in cols.items()})
    out["dico"] = d["dico"]
    m = out.groupby("dico").sum()
    missing = set(m.index) - set(N2)
    assert not missing, missing
    return m


def munis(label, scheme):
    """Set of DICOs for a pollster region label; None if the region cannot be mapped."""
    l = label.lower()
    allm = set(N2)
    n2 = lambda c: {k for k, v in N2.items() if v == c}
    n3 = lambda *c: {k for k, v in N3.items() if v in c}
    amp, aml = n3("11A"), n3("170")
    if scheme == "nuts2024":
        if l == "norte":
            return n2("11")
        if l == "centro":
            return n2("16") - n3("16B", "16I")
        if "oeste" in l:
            return n3("16B", "16I", "185")
        if l == "grande lisboa":
            return aml - SETUBAL_SOUTH_AML
        if "setúbal" in l or "setubal" in l:
            return SETUBAL_SOUTH_AML
        if l == "alentejo":
            return n2("18") - n3("185")
        if l == "algarve":
            return n2("15")
        if "açores" in l or "acores" in l:
            return n2("20")
        if "madeira" in l:
            return n2("30")
        return None
    if scheme == "nonstandard":  # Marktest / Intercampus 2011-2015 regions: the old metropolitan NUTS-III
        if re.search(r"gr(ande|\.) ?porto", l):
            return OLD_GR_PORTO
        if re.search(r"gr(ande|\.) ?lisb", l):
            return OLD_GR_LISBOA
    split_porto = scheme == "split_porto"
    if re.search(r"am porto|^amp$|grande porto|gr\. porto", l):
        return amp
    if re.search(r"am lisboa|^aml$|^lisboa$|grande lisboa|gr\. lisb", l):
        return aml
    if l == "norte":
        return n2("11") - amp if split_porto else n2("11")
    if l == "centro":
        return n2("16")
    if l == "alentejo":
        return n2("18")
    if l == "algarve":
        return n2("15")
    if re.fullmatch(r"sul (e|\+) ilhas", l):
        return n2("18") | n2("15") | n2("20") | n2("30")
    if l == "sul e alentejo":
        return None
    if l == "sul":
        return n2("18") | n2("15") if scheme in ("split_porto",) else None
    if l == "ilhas":
        return n2("20") | n2("30")
    if "açores" in l:
        return n2("20")
    if "madeira" in l:
        return n2("30")
    return None


EXTRA_EXCL = re.compile(r"vencer|nunca votaria|sente mais pr[óo]ximo|desigualdade|tarifa", re.I)
PARTY = [
    ("PS", re.compile(r"^PS$")),
    (
        "AD",
        re.compile(
            r"^(PSD|AD|PPD/PSD|PPD|PSD/PPD|PSD/CDS.*|PSD\.CDS.*|PORTUGAL À FRENTE|PAF|PÀF|ALIANÇA DEMOCRÁTICA.*|NOVA AD.*|CDS|CDS/PP|CDS-PP)$",
            re.I,
        ),
    ),
    ("CH", re.compile(r"^(CH|CHEGA)$", re.I)),
    ("IL", re.compile(r"^(IL|INICIATIVA LIBERAL)$", re.I)),
    ("BE", re.compile(r"^(BE|B\.E\.|BLOCO( DE)? ESQUERDA)$", re.I)),
    ("CDU", re.compile(r"^(CDU|PCP|PCP/CDU|CDU \(PCP/VERDES.*|PCP-PEV)$", re.I)),
    ("L", re.compile(r"^(L|LIVRE|PARTIDO LIVRE/TEMPO DE AVANÇAR)$", re.I)),
    ("PAN", re.compile(r"^PAN$", re.I)),
]
ELECTIONS = ["2011-06-05", "2015-10-04", "2022-01-30", "2024-03-10", "2025-05-18"]


def clean(c):
    c = str(c)
    n = re.search(r"base:\s*(\d+)", c, re.I)
    c = re.sub(r"\n.*", "", c, flags=re.S)
    c = re.sub(r"^(regi[ãa]o(es|ões)?\s*(\(c\.?3\))?\s*[-:]?\s*)", "", c.strip(), flags=re.I).strip()
    return c, (int(n.group(1)) if n else None)


def scheme_of(labels):
    L = " ".join(labels).lower()
    if re.search(r"litor|inter", L):
        return "nonstandard"
    if re.search(r"oeste|pen[íi]nsula", L):
        return "nuts2024"
    if re.search(r"am porto|\bamp\b|grande porto|gr\. porto", L):
        return "split_porto"
    return "nuts2"


def election_of(date):
    for e in ELECTIONS:
        if 0 < (dt.date.fromisoformat(e) - date).days <= 56:
            return e


def load():
    out, notes = [], []
    for f in sorted(glob.glob(os.path.join(PATHS["deposits"], "*.json"))):
        d = json.load(open(f))
        num = d["deposit_number"]
        date = dt.date.fromisoformat(DEP[num]["deposit_datetime"][:10])
        e = election_of(date)
        if e is None:
            continue
        cs = [i for i in candidates(d) if not EXTRA_EXCL.search(label(d["matrices"][i]))]
        if not cs:
            notes.append((num, "no vote-intention table"))
            continue
        m = d["matrices"][cs[0]]
        cols = [clean(c) for c in m["column_labels"]]
        rows = m["rows"]
        nv = max(len(r.get("values") or []) for r in rows)
        last_empty = all(len(r.get("values") or []) < nv or r["values"][-1] is None for r in rows)
        if (
            last_empty
            and not re.search(r"global|total", cols[0][0], re.I)
            and munis(cols[0][0], "nuts2") is None
        ):
            # a stub label with no value of its own: every value sits one label to the left
            cols = cols[1:]
            rows = [dict(r, values=(r.get("values") or [])[:-1]) for r in rows]
            nv -= 1
        if nv == len(cols) - 1 and not re.search(r"global|total", cols[1][0], re.I) is None:
            cols = cols[1:]  # a stray stub label before the value columns
        if nv != len(cols):
            notes.append((num, f"values {nv} vs columns {len(cols)}"))
            continue
        names = [c for c, _ in cols]
        scheme = scheme_of(names)
        tot = [j for j, c in enumerate(names) if re.fullmatch(r"total|global", c, re.I)]
        regs = [(j, c, n) for j, (c, n) in enumerate(cols) if munis(c, scheme) is not None]
        dropped = [
            c
            for c in names
            if re.search(r"litor|interior|^sul$|sul e alentejo", c, re.I) and munis(c, scheme) is None
        ]
        vals, ad_parts, ad_coalition = {}, {}, None
        SINGLE = re.compile(r"^(PSD|PPD/PSD|PPD|PSD/PPD|CDS|CDS/PP|CDS-PP)$", re.I)
        for r in rows:
            nm = str(r.get("party_normalized") or r.get("party_raw") or "").strip()
            for p, rx in PARTY:
                if rx.match(nm):
                    v = np.array([np.nan if x is None else float(x) for x in r["values"]])
                    if p == "AD":
                        if SINGLE.match(nm):
                            ad_parts.setdefault(nm.upper().split("/")[0].replace("PPD", "PSD"), v)
                        elif ad_coalition is None:
                            ad_coalition = v
                    elif p not in vals:
                        vals[p] = v
        if ad_coalition is not None:
            vals["AD"] = ad_coalition
        elif ad_parts:
            vals["AD"] = sum(ad_parts.values())  # PSD and CDS listed apart: the coalition is their sum
        if tot:
            T0 = np.array([v[tot[0]] for v in vals.values()])
            stub = [
                j
                for j in range(tot[0] + 1, nv)
                if np.allclose(np.array([v[j] for v in vals.values()]), T0, equal_nan=True)
            ]
            if stub:
                notes.append((num, f"columns {stub} repeat the total: values shifted, dropped"))
                continue
        # share of the table's respondents who name one of the parties (raw tables keep undecided rows)
        allrows = [np.array([np.nan if x is None else float(x) for x in r["values"]]) for r in rows]
        tc = tot[0] if tot else None
        named = None
        if tc is not None:
            whole = np.nansum([a[tc] for a in allrows])
            part = np.nansum([v[tc] for v in vals.values()])
            named = part / whole if whole > 0 else None
        base = re.search(r"(\d+)\s*%\s*do total", label(m))
        if base and named is not None:
            named *= int(base.group(1)) / 100
        if "PS" not in vals or "AD" not in vals or len(vals) < 4:
            notes.append((num, "fewer than four parties"))
            continue
        out.append(
            dict(
                deposit=num,
                company=DEP[num]["company"].strip(' "*').title(),
                date=date,
                election=e,
                matrix=cs[0],
                heading=label(m)[:80],
                scheme=scheme,
                total_col=tot[0] if tot else None,
                regions=regs,
                dropped=dropped,
                values=vals,
                named=named,
            )
        )
    return out, notes


MAJOR = 0.05  # parties with at least this share (among the eight) nationally
NAMES = {"Duplimétrica": "Duplimétrica/IPESPE", "Metris": ("Metris", "ICS/ISCTE/GFK Metris")}


def match_sizes(tabs):
    """n, fieldwork days and new interviews per table, from the poll list."""
    used, out = set(), {}
    for t in sorted(tabs, key=lambda x: x["date"]):
        names = NAMES.get(t["company"], t["company"])
        names = names if isinstance(names, tuple) else (names,)
        c = polls[
            polls.Pollster.isin(names)
            & (polls["Fieldwork End"] <= t["date"])
            & (polls["Fieldwork End"] >= t["date"] - dt.timedelta(days=12))
        ]
        c = c[~c.index.isin(used)].sort_values("Fieldwork End")
        if len(c):
            r = c.iloc[-1]
            used.add(r.name)
            out[t["deposit"]] = dict(
                n=int(r["Sample Size"]), start=r["Fieldwork Start"], end=r["Fieldwork End"], matched=True
            )
        else:
            out[t["deposit"]] = dict(
                n=None, start=t["date"] - dt.timedelta(days=4), end=t["date"], matched=False
            )
    return out


def truth_shares(year):
    v = party_votes(year)
    return v


def analyse(tabs):
    sizes = match_sizes(tabs)
    # fill unmatched with the pollster-election median
    med = collections.defaultdict(list)
    for t in tabs:
        if sizes[t["deposit"]]["n"]:
            med[(t["company"], t["election"])].append(sizes[t["deposit"]]["n"])
    for t in tabs:
        s = sizes[t["deposit"]]
        if s["n"] is None:
            s["n"] = int(np.median(med[(t["company"], t["election"])] or [800]))
    # new interviews: fieldwork days not covered by the same pollster's earlier tables
    seen = collections.defaultdict(set)
    for t in sorted(tabs, key=lambda x: sizes[x["deposit"]]["start"]):
        s = sizes[t["deposit"]]
        k = (t["company"], t["election"])
        days = {s["start"] + dt.timedelta(days=i) for i in range((s["end"] - s["start"]).days + 1)}
        s["new"] = s["n"] * len(days - seen[k]) / len(days)
        s["days"] = days
        seen[k] |= days
    votes = {e: truth_shares(e[:4]) for e in ELECTIONS}
    cells, checks = [], []
    for t in tabs:
        v = votes[t["election"]]
        s = sizes[t["deposit"]]
        parties = [p for p in PARTIES if p in t["values"]]
        cols = [j for j, _, _ in t["regions"]] + ([t["total_col"]] if t["total_col"] is not None else [])
        # a blank region cell under a printed total is a share that rounds to zero
        vals = {p: np.where(np.isnan(t["values"][p]), 0.0, t["values"][p]) for p in parties}
        parties = [
            p for p in parties if t["total_col"] is None or np.isfinite(t["values"][p][t["total_col"]])
        ]
        regs = [(j, lab, munis(lab, t["scheme"])) for j, lab, _ in t["regions"]]
        islands = any(m & ISLANDS for _, _, m in regs)
        U = set(N2) if islands else MAINLAND
        tv = v.loc[v.index.isin(U), parties]
        true_U = tv.sum().values / tv.values.sum()
        major = true_U >= MAJOR
        P = np.array([[vals[p][j] for p in parties] for j, _, _ in regs], float)
        P = P / P.sum(1, keepdims=True)
        w = np.array([v.loc[v.index.isin(m & U), parties].values.sum() for _, _, m in regs]) / tv.values.sum()
        covered = abs(w.sum() - 1) < 0.02
        wmean = (w[:, None] * P).sum(0) / w.sum()
        if t["total_col"] is not None:
            tot = np.array([vals[p][t["total_col"]] for p in parties], float)
            tot = tot / tot.sum()
            if covered:
                checks.append(
                    dict(
                        deposit=t["deposit"],
                        company=t["company"],
                        election=t["election"],
                        maxdiff_pp=100 * np.abs(tot - wmean).max(),
                    )
                )
        else:
            if not covered:
                continue
            tot = wmean
        for (j, lab, m), pr, wr in zip(regs, P, w):
            tr = v.loc[v.index.isin(m & U), parties]
            true_r = tr.sum().values / tr.values.sum()
            for k, p in enumerate(parties):
                if not major[k] or pr[k] <= 0:
                    continue
                named = t["named"] if t["named"] else 1.0
                n_r, n = s["n"] * named * wr, s["n"] * named
                cells.append(
                    dict(
                        deposit=t["deposit"],
                        company=t["company"],
                        election=t["election"],
                        region=lab,
                        rkey=frozenset(m & U),
                        party=p,
                        w=wr,
                        n=n,
                        n_r=n_r,
                        new=s["new"] * named,
                        e=np.log(pr[k] / tot[k]) - np.log(true_r[k] / true_U[k]),
                        v=(1 - true_r[k]) / true_r[k] * (1 / n_r - 1 / n),
                        start=s["start"],
                        end=s["end"],
                        share=true_r[k],
                    )
                )
    return pd.DataFrame(cells), pd.DataFrame(checks), sizes


def deff_estimate(c, by_pair=False):
    """Observed scatter between non-overlapping tables of the same pollster-election, over the binomial."""
    num, den, pairs = 0.0, 0.0, 0
    out = []
    for (co, e), g in c.groupby(["company", "election"]):
        deps = g.drop_duplicates("deposit").sort_values("start")
        chosen, last_end = [], None
        for _, r in deps.iterrows():
            if last_end is None or r["start"] > last_end:
                chosen.append(r["deposit"])
                last_end = r["end"]
        for a, b in zip(chosen[:-1], chosen[1:]):
            A = g[g.deposit == a].set_index(["rkey", "party"])
            B = g[g.deposit == b].set_index(["rkey", "party"])
            idx = A.index.intersection(B.index)
            a_, b_ = ((A.loc[idx, "e"] - B.loc[idx, "e"]) ** 2).sum(), (
                A.loc[idx, "v"] + B.loc[idx, "v"]
            ).sum()
            num += a_
            den += b_
            pairs += 1
            out.append((co, e, a_, b_))
    if by_pair:
        return out
    return num / den if den else np.nan, pairs


def pollster_bias(c, deff):
    """Per pollster-election: rms of the pooled regional error, minus its sampling variance."""
    rows = []
    for (co, e), g in c.groupby(["company", "election"]):
        N = g.drop_duplicates("deposit")["new"].sum()
        agg = []
        for (rk, p), h in g.groupby(["rkey", "party"]):
            lab = h["region"].iloc[0]
            wts = h["new"].values
            eb = (wts * h["e"].values).sum() / wts.sum()
            s, w = h["share"].iloc[0], h["w"].iloc[0]
            var = deff * (1 - s) / s * (1 / (N * w) - 1 / N)
            agg.append((lab, p, eb, var, h["rkey"].iloc[0], s))
        a = pd.DataFrame(agg, columns=["region", "party", "eb", "var", "rkey", "share"])
        b2 = (a.eb**2 - a["var"]).mean()
        rows.append(
            dict(
                company=co,
                election=e,
                tables=g.deposit.nunique(),
                distinct_n=round(N),
                regions=g.rkey.nunique(),
                raw_rms=np.sqrt((a.eb**2).mean()),
                sampling_rms=np.sqrt(a["var"].mean()),
                bias=np.sqrt(max(b2, 0)),
                bias2=b2,
                cells=len(a),
                agg=a,
            )
        )
    return rows


def weighted_bias(aggs, b0=0.07):
    """Bias rms over cells, each cell's (eb^2 - var) weighted by 1 / (var + b0^2)^2."""
    a = pd.concat(aggs)
    w = 1 / (a["var"] + b0**2) ** 2
    b2 = (w * (a.eb**2 - a["var"])).sum() / w.sum()
    return np.sqrt(max(b2, 0)), b2, len(a)


def boot_bias(rows, reps=2000, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(reps):
        pick = rng.integers(0, len(rows), len(rows))
        out.append(weighted_bias([rows[i]["agg"] for i in pick])[1])
    q = np.quantile(out, [0.05, 0.95])
    return np.sqrt(np.clip(q, 0, None))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--deposits", required=True, help="the v2 parser's output folder (one JSON per deposit)")
    ap.add_argument("--deposits-csv", required=True, help="the ERC deposit list (deposits.csv)")
    ap.add_argument("--nuts", required=True, help="municipality to NUTS-III/II (2013) parquet")
    ap.add_argument(
        "--deff",
        type=float,
        default=None,
        help="design effect for the sampling correction (default: measured, at least 1)",
    )
    ap.add_argument("--out", default=None, help="write the per-table cells to this CSV")
    args = ap.parse_args(argv)
    setup(args.deposits, args.deposits_csv, args.nuts)
    tabs, notes = load()
    c, checks, sizes = analyse(tabs)
    unmatched = [d for d, s in sizes.items() if not s["matched"]]
    print("tables", len(tabs), "cells", len(c), "unmatched sample sizes", unmatched)
    print("\nconsistency: total column vs region-weighted mean (max pp over parties)")
    print(checks.sort_values("maxdiff_pp", ascending=False).head(12).to_string(index=False))
    deff, npairs = deff_estimate(c)
    print(
        f"\nmeasured design effect (non-overlapping pairs, includes campaign drift): {deff:.2f} over {npairs} pairs"
    )
    pp = pd.DataFrame(deff_estimate(c, by_pair=True), columns=["company", "election", "num", "den"])
    print(
        pp.groupby(["company", "election"]).sum().assign(ratio=lambda d: d.num / d.den).round(2).to_string()
    )
    rng = np.random.default_rng(0)
    bs = [pp.iloc[rng.integers(0, len(pp), len(pp))] for _ in range(2000)]
    print(
        "deff 90% interval over pairs:",
        np.round(np.quantile([b.num.sum() / b.den.sum() for b in bs], [0.05, 0.95]), 2),
    )
    D = args.deff if args.deff else max(deff, 1.0)
    print("design effect used for the sampling correction:", D)
    rows = pollster_bias(c, D)
    print("\nper pollster-election (log scale, parties >=5%):")
    print(
        pd.DataFrame([{k: v for k, v in r.items() if k != "agg"} for r in rows])
        .round(3)
        .to_string(index=False)
    )
    print(
        "\nper election: interviews naming a party (distinct), effective at deff 1 and at the measured deff, bias"
    )
    for e in ELECTIONS:
        rs = [r for r in rows if r["election"] == e]
        N = sum(r["distinct_n"] for r in rs)
        b, _, k = weighted_bias([r["agg"] for r in rs])
        lo, hi = boot_bias(rs) if len(rs) > 1 else (np.nan, np.nan)
        print(
            f"{e}: pollsters {len(rs)}, named interviews {N}, effective {N:.0f} (deff 1) / {N / deff:.0f} (deff {deff:.2f}), "
            f"bias {b:.3f} [90% {lo:.3f}-{hi:.3f}] over {k} cells"
        )
    ok = [r for r in rows if not r["election"].startswith("2022")]
    b, _, k = weighted_bias([r["agg"] for r in ok])
    lo, hi = boot_bias(ok)
    print(f"all pollster-elections but 2022: bias {b:.3f} [90% {lo:.3f}-{hi:.3f}] over {k} cells")
    for thr in (0.10,):
        sub = [dict(r, agg=r["agg"][r["agg"].share >= thr]) for r in ok]
        sub = [r for r in sub if len(r["agg"])]
        b, _, k = weighted_bias([r["agg"] for r in sub])
        lo, hi = boot_bias(sub)
        print(
            f"  parties with >= {thr:.0%} in the region: bias {b:.3f} [90% {lo:.3f}-{hi:.3f}] over {k} cells"
        )
    # pooled across pollsters: same region (same municipalities), same party
    print("\npooled across pollsters, regions shared by 2+ pollsters:")
    allagg = pd.concat(
        [r["agg"].assign(company=r["company"], election=r["election"], N=r["distinct_n"]) for r in rows]
    )
    for e, g in allagg.groupby("election"):
        out = []
        for (rk, p), h in g.groupby(["rkey", "party"]):
            if h.company.nunique() < 2:
                continue
            wt = 1 / (h["var"] + 1e-6)
            eb = (wt * h.eb).sum() / wt.sum()
            var = 1 / wt.sum()
            out.append((h.region.iloc[0], p, eb, var, h.company.nunique()))
        if out:
            o = pd.DataFrame(out, columns=["region", "party", "eb", "var", "k"])
            print(
                f"  {e}: cells {len(o)}, raw rms {np.sqrt((o.eb**2).mean()):.3f}, sampling rms {np.sqrt(o['var'].mean()):.3f}, "
                f"bias {np.sqrt(max((o.eb**2 - o['var']).mean(), 0)):.3f}"
            )
            print("    ", "; ".join(f"{r.region} {r.party} {r.eb:+.2f}" for r in o.itertuples()))
    # persistence of a pollster's regional error across elections
    print("\npersistence (same pollster, same region and party, consecutive elections):")
    for co, g in allagg.groupby("company"):
        es = sorted(g.election.unique())
        for a, b in zip(es[:-1], es[1:]):
            A = g[g.election == a].set_index(["rkey", "party"]).eb
            B = g[g.election == b].set_index(["rkey", "party"]).eb
            idx = A.index.intersection(B.index)
            if len(idx) >= 4:
                print(
                    f"  {co} {a[:4]}->{b[:4]}: cells {len(idx)}, corr {np.corrcoef(A[idx], B[idx])[0, 1]:+.2f}"
                )
    if args.out:
        c.drop(columns=["rkey"]).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
