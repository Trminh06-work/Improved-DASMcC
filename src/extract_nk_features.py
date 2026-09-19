"""neurokit2 feature extraction, in one of three readings of the 12-lead record.

    python extract_nk_features.py {timemajor,leadmajor,perlead}

The sixteen features are identical throughout; only what the delineator is run
on changes, which changes how many beats it finds and how the columns are named:

    timemajor   one pass over the 12000 vector with the leads interleaved   16 cols
    leadmajor   one pass over the 12000 vector with the leads end to end    16 cols
    perlead     twelve passes, one per 1000-sample lead                    192 cols

Writes ../data/features_<ordering>.parquet, one row per record, each cell a variable-length series.
"""

import argparse
import numpy as np
import pandas as pd
import neurokit2 as nk
from joblib import Parallel, delayed
from pathlib import Path
from tqdm.auto import tqdm

DATA = Path(__file__).resolve().parent.parent / "data"

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
SAMPLING_RATE = 100
WAVES = ["ECG_P_Peaks", "ECG_Q_Peaks", "ECG_S_Peaks", "ECG_T_Peaks"]

# Each takes one (12, 1000) record and returns the signals to delineate.
SEGMENTS = {
    "timemajor": lambda record: [record.T.reshape(-1)],
    "leadmajor": lambda record: [record.reshape(-1)],
    "perlead":   lambda record: list(record),
}



def amplitude(signal, index):
    """Signal value at each peak index, NaN where the peak was not found."""
    out = np.full(len(index), np.nan, dtype = np.float32)
    found = ~np.isnan(index)
    out[found] = signal[index[found].astype(int)]
    return out


def extract(signal):
    """The sixteen features for one signal, however long."""
    try:
        _, info = nk.ecg_peaks(signal, sampling_rate = SAMPLING_RATE)
        R = np.asarray(info["ECG_R_Peaks"], dtype = np.float32)
    except Exception:
        R = np.array([], dtype = np.float32)

    try:
        _, waves = nk.ecg_delineate(signal, R, sampling_rate = SAMPLING_RATE,
                                    method = "peak")
        P, Q, S, T = (np.asarray(waves[k], dtype = np.float32) for k in WAVES)
    except Exception:
        P = Q = S = T = np.full(len(R), np.nan, dtype = np.float32)

    return {
        "P_loc": P, "Q_loc": Q, "R_loc": R, "S_loc": S, "T_loc": T,
        "P_amp": amplitude(signal, P), "Q_amp": amplitude(signal, Q),
        "R_amp": amplitude(signal, R), "S_amp": amplitude(signal, S),
        "T_amp": amplitude(signal, T),
        "PQ": Q - P, "ST": T - S, "QT": T - Q, "PR": R - P, "QRS": S - Q,
        "RR": np.diff(R, prepend = np.nan).astype(np.float32),
    }


def row(record, ordering):
    """One record's features; per-lead columns get a lead prefix."""
    segments = SEGMENTS[ordering](record)
    if len(segments) == 1:
        return extract(segments[0])
    return {f"{lead}_{key}": values
            for lead, segment in zip(LEADS, segments)
            for key, values in extract(segment).items()}



def main():
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("ordering", choices = list(SEGMENTS))
    ordering = parser.parse_args().ordering

    out = DATA / f"features_{ordering}.parquet"
    if out.exists():
        raise SystemExit(f"{out.name} already exists; delete it to re-extract")

    df = pd.read_parquet(DATA / "ptbxl100_single.parquet")
    records = np.stack(df["signal"].values).reshape(len(df), 1000, 12).transpose(0, 2, 1)

    jobs = Parallel(n_jobs = -1, return_as = "generator")(
        delayed(row)(record, ordering) for record in records)
    rows = list(tqdm(jobs, total = len(records), desc = ordering))

    features = pd.DataFrame(rows, index = df.index)
    features["label"] = df["diagnostic_superclass"]
    features.to_parquet(out)

    peaks = next(c for c in features.columns if c.endswith("R_loc"))
    counts = features[peaks].map(len)
    print(f"{peaks}: median {int(counts.median())}  min {counts.min()}  max {counts.max()}")
    print(f"saved {out}  {features.shape}")


if __name__ == "__main__":
    main()
