import pandas as pd
import numpy as np
import wfdb
import ast
import json

from pathlib import Path
path = str(Path(__file__).resolve().parent.parent / 'data') + '/'


def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data], dtype = np.float32)
    return data


def retrieve():
    def aggregate_diagnostic(y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key].diagnostic_class)
        return list(set(tmp))

    sampling_rate=100

    # load and convert annotation data
    Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

    # Load raw signal data
    X = load_raw_data(Y, sampling_rate, path)

    # Load scp_statements.csv for diagnostic aggregation
    agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]


    # Apply diagnostic superclass
    Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

    return X, Y


def combine(X, Y):
    """One row per ECG: patient_id, diagnostic_superclass, and the flattened signal.

    Signals are flattened to 12000 values (1000 timesteps x 12 leads); recover the
    original shape with np.stack(df.signal.values).reshape(-1, 1000, 12).
    """
    assert len(X) == len(Y), f"X has {len(X)} records but Y has {len(Y)}"

    return pd.DataFrame({
        'patient_id': Y['patient_id'].values,
        'diagnostic_superclass': Y['diagnostic_superclass'].values,
        'signal': list(X.reshape(len(X), -1)),
    }, index = Y.index)
