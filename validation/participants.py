"""Load the DIC Challenge 2.0 participants' centre-row line cuts.

FORMAT (as distributed)
    Subset,9,13,17,...,49,,9,13,17,...,49
    Pixel,Noise,Noise,...,Noise,,Deformed,Deformed,...,Deformed
    1,NaN,NaN,...
    2,...

    Two blocks side by side, the noise-floor image and the deformed image, with
    one column per length parameter, separated by an empty column. Pixel is
    1-based. The first header cell names the parameter ("Subset" for local
    codes; global codes put element sizes there), so it is read, not assumed.

    Parameter lists differ between codes (Local.Affine.B uses 9..49 in steps
    of 4, not the paper's "9 to 59 in increments of 10"), so they are read per
    file. Files marked _None were not submitted for that quantity and may be
    empty; they load as an empty frame rather than raising.
"""

import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd


def _to_float(cell: str) -> float:
    cell = cell.strip()
    if not cell or cell.lower() == "nan":
        return np.nan
    try:
        return float(cell)
    except ValueError:
        return np.nan


MAX_PIXEL_STEP = 50  # larger jumps mean a broken pixel column, not omitted rows
_NUMBER = re.compile(r"(\d+(?:\.\d+)?)")
_DEFORMED_TOKENS = ("deform", "disp", "strain", "eyy", "exx", "exy")


def _kind(label: str) -> str | None:
    """'noise', 'deformed' or None, from a free-text column label.

    Participants wrote 'Noise', 'v-noise', 'v noise', 'displ-y noise',
    'Subset_09_vNoise' for the noise-floor image, and 'Deformed', 'Displ.',
    'v-disp', 'displ-y', 'Subset_09_vdeformed' for the deformed one. 'noise'
    is checked first because some noise labels also contain 'displ'.
    """
    label = label.lower()
    if "noise" in label:
        return "noise"
    if any(t in label for t in _DEFORMED_TOKENS):
        return "deformed"
    return None


def load_participant(path: Path) -> pd.DataFrame:
    """Long frame: code, parameter_label, parameter, config, image, kind_source, pixel, value.

    ANCHOR  The row whose first cell is 'Pixel', present in every submission.
            Up to two header rows above it are read; any preamble before that
            is ignored.

    PAIRING The i-th noise column pairs with the i-th deformed column (`config`).
            Labels are not used for pairing: some files give fewer parameter
            labels than columns (Global.B strain), so grouping by label would
            merge different runs. `parameter` is kept for display only; MEI
            does not depend on it.

    PARAMETER First number in the row above 'Pixel' ('9', 'lc 9', 'subset_09',
            'SS = 8, SW = 5'), else in the 'Pixel' row itself ('R=1px'; VSG pairs
            33,33). The order matters: Pixel-row labels can carry digits that are
            not parameters (Misc.F's 'e22-Noise' names the strain component).
            A blank inherits from the left. Display only.

    POSITION From the file's pixel column when it is strictly increasing in steps
            of at most MAX_PIXEL_STEP: that covers 0-based files and files that omit
            rows, where the column carries real positions. Otherwise the column is
            broken (rounded to three significant figures, a duplicated row, a typo)
            and positions fall back to row order anchored at the file's first pixel,
            marked pixel_repaired=True.

    KIND    From the 'Pixel' row and the row above ('Noise', 'v-deformed',
            'Displ.'), else two rows above ('V-vNoise' in Local.Affine.H). An
            unlabelled column inherits the kind of its block, reset by a blank
            separator column. If nothing is labelled but parameters come in
            identical adjacent pairs (Local.Quad.D), each pair is resolved from
            the data: the deformed column has the larger long-period plateau,
            since the noise floor averages to zero. Marked kind_source='inferred'.
    """
    path = Path(path)
    columns_out = [
        "code",
        "parameter_label",
        "parameter",
        "config",
        "image",
        "kind_source",
        "pixel",
        "value",
        "pixel_repaired",
    ]
    empty = pd.DataFrame(columns=columns_out)
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as handle:
        rows = list(csv.reader(handle))

    p = next(
        (
            i
            for i, r in enumerate(rows)
            if r and r[0].strip().lower().startswith("pixel")
        ),
        None,
    )
    if p is None:
        return empty

    data_rows = [r for r in rows[p + 1 :] if r and not np.isnan(_to_float(r[0]))]
    if not data_rows:
        return empty
    width = max(len(r) for r in [rows[p], *(rows[max(0, p - 2) : p]), *data_rows[:50]])
    pad = lambda r: r + [""] * (width - len(r))
    pix_row = pad(rows[p])
    above = pad(rows[p - 1]) if p >= 1 else [""] * width
    above2 = pad(rows[p - 2]) if p >= 2 else [""] * width
    values = np.array([[_to_float(c) for c in pad(r)[:width]] for r in data_rows])
    file_pixels = values[:, 0]
    steps = np.diff(file_pixels)
    trustworthy = (
        file_pixels.size > 1 and np.all(steps > 0) and np.all(steps <= MAX_PIXEL_STEP)
    )
    if trustworthy:
        # Offsets (0-based files) and omitted rows (e.g. invalid edges left out) are fine:
        # the column carries real positions. Normalise to 1-based.
        pixels = (
            (file_pixels - file_pixels.min() + 1).astype(int)
            if file_pixels.min() < 1
            else file_pixels.astype(int)
        )
        repaired = False
    else:
        # Broken column, consecutive rows. Seen: pixel numbers rounded to three significant
        # figures from 1000 up (Local.Affine.D, Local.Quad.B), one duplicated row
        # (Local.Affine.G), one typo (Global.B strain: 3999 -> 40000). The rows are still
        # one per image column, so positions follow row order, anchored at the file's
        # first pixel, which is exact (below 1000) in every case seen. Starting at 1
        # instead would shift files that begin at pixel 7 or 17 by that many columns.
        first = file_pixels[0]
        start = int(first) if np.isfinite(first) and 0 <= first < len(data_rows) else 1
        pixels = np.arange(start, start + len(data_rows))
        if start == 0:
            pixels = pixels + 1
        repaired = True
    label = (above[0].strip() or "parameter") if p >= 1 else "parameter"

    cols, last_kind, last_param = [], None, np.nan
    for j in range(1, width):
        header_blank = not (pix_row[j].strip() or above[j].strip() or above2[j].strip())
        if header_blank and np.isnan(values[:, j]).all():
            last_kind = None  # separator between blocks
            continue
        m = _NUMBER.search(above[j]) or _NUMBER.search(pix_row[j])
        param = float(m.group(1)) if m else last_param
        kind = _kind(f"{above[j]} {pix_row[j]}") or _kind(above2[j])
        source = "label"
        if kind is None and last_kind is not None:
            kind, source = last_kind, "inherited"
        cols.append([j, param, kind, source])
        last_kind = kind if kind else last_kind
        last_param = param

    if not any(c[2] for c in cols):
        far = pixels >= 0.8 * pixels.max()
        for a, b in zip(cols, cols[1:]):
            if a[2] is None and b[2] is None and np.isfinite(a[1]) and a[1] == b[1]:
                pa = np.nanmedian(np.abs(values[far, a[0]]))
                pb = np.nanmedian(np.abs(values[far, b[0]]))
                a[2], b[2] = ("deformed", "noise") if pa > pb else ("noise", "deformed")
                a[3] = b[3] = "inferred"

    cols = [c for c in cols if c[2]]
    ordinal = {"noise": 0, "deformed": 0}
    records = []
    for j, param, kind, source in cols:
        config = ordinal[kind]
        ordinal[kind] += 1
        for pix, v in zip(pixels, values[:, j]):
            records.append(
                (path.stem, label, param, config, kind, source, int(pix), v, repaired)
            )
    return pd.DataFrame(records, columns=columns_out) if records else empty


def load_folder(folder: Path) -> pd.DataFrame:
    """Every participant CSV in a results folder, concatenated."""
    frames = [load_participant(p) for p in sorted(Path(folder).glob("*.csv"))]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def code_plateaus(table: pd.DataFrame) -> pd.Series:
    """Each code's long-period plateau: per configuration, median |deformed value|
    over the last 20 % of pixels (longest periods, least attenuation); then the
    median over configurations, so one odd column cannot decide a code's units."""
    out = {}
    for code, g in table[table.image == "deformed"].groupby("code"):
        per_config = []
        for _, c in g.groupby("config"):
            far = c.loc[c.pixel >= 0.8 * c.pixel.max(), "value"].abs()
            if far.notna().any():
                per_config.append(far.median())
        out[code] = float(np.median(per_config)) if per_config else np.nan
    return pd.Series(out, name="plateau", dtype=float)


def classify_quantity(table: pd.DataFrame, tolerance: float = 0.25) -> str:
    """Vote per code: which commanded amplitude does its plateau match, in a known unit?

    Each code votes for displacement (plateau ~0.5 px) or strain (~0.05, or in
    percent / microstrain). Codes matching neither abstain. A vote rather than a
    median, so a minority of codes in odd units cannot flip the folder. Used only
    to cross-check the folder name.
    """

    def matches(ratio: float) -> bool:
        return np.isfinite(ratio) and any(
            abs(ratio / f - 1) < tolerance for f in KNOWN_UNITS
        )

    votes = {"displacement": 0, "strain": 0}
    for plateau in code_plateaus(table):
        votes["displacement"] += matches(plateau / 0.5)
        votes["strain"] += matches(plateau / 0.05)
    return max(votes, key=votes.get)


KNOWN_UNITS = {1.0: "as submitted", 100.0: "percent", 1e6: "microstrain"}


def harmonise_units(table: pd.DataFrame, amplitude: float, tolerance: float = 0.25):
    """Rescale codes reported in a known unit; exclude codes matching none.

    A code's plateau divided by the commanded amplitude should be about 1. A
    ratio near 100 or 1e6 is percent or microstrain and is rescaled. Anything
    else, e.g. a displacement submission in the strain folder (ratio ~10), is
    excluded, since it would add a meaningless entry to the comparison.
    Returns (harmonised_table, report).
    """
    rows, factors = [], {}
    for code, plateau in code_plateaus(table).items():
        ratio = plateau / amplitude
        factor = next(
            (
                f
                for f in KNOWN_UNITS
                if np.isfinite(ratio) and abs(ratio / f - 1) < tolerance
            ),
            None,
        )
        factors[code] = factor
        rows.append(
            {
                "code": code,
                "plateau": plateau,
                "ratio_to_amplitude": ratio,
                "unit": KNOWN_UNITS.get(factor, "EXCLUDED: no known unit"),
            }
        )
    keep = [c for c, f in factors.items() if f is not None]
    out = table[table.code.isin(keep)].copy()
    out["value"] = out["value"] / out["code"].map(factors)
    return out, pd.DataFrame(rows)


_NEW_SUFFIX = re.compile(r"_?New$", re.IGNORECASE)


def drop_superseded(table: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Keep resubmissions, drop what they replace, and name them as the original.

    Some participants resubmitted ('Misc.C_New' in one folder, 'Misc.CNew' in
    the other). The paper's figures use the resubmission under the original
    name, so that is what is returned. Also returns the codes that were dropped.
    """
    codes = set(table.code.unique())
    rename, dropped = {}, []
    for code in codes:
        base = _NEW_SUFFIX.sub("", code)
        if base != code:
            rename[code] = base
            if base in codes:
                dropped.append(base)
    out = table[~table.code.isin(dropped)].copy()
    out["code"] = out["code"].replace(rename)
    return out, sorted(set(dropped))


def not_submitted(folder: Path) -> list[str]:
    """Codes whose file is marked _None: nothing was submitted for this quantity."""
    return sorted(p.stem.replace("_None", "") for p in Path(folder).glob("*_None.csv"))
