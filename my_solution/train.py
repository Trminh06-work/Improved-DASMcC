"""Train one model per invocation, as listed in job.conf.

The CNN trains on the raw (12, 1000) leads. The autoencoder trains on the
neurokit features concatenated with the CNN's 125 extracted features, so it
needs a finished CNN checkpoint -- hence the two stages in job.conf.

Training only: both models fit on the 80 % split and report train loss. The held-out
20 % is evaluated in my_notebook.ipynb, where the classifiers are scored.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from Autoencoder import Autoencoder
from CNN_extractor import CNNModel, CNNFeatureExtractor

DATA = Path(__file__).resolve().parent.parent / "data"
LABEL_ENCODING = {"NORM": 0, "MI": 1, "STTC": 2, "HYP": 3, "CD": 4}
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def load_signals():
    """(N, 12, 1000) lead-major signals, and labels encoded as in Table 5."""
    df = pd.read_parquet(DATA / "ptbxl100_single.parquet")

    # Stored flattened time-major, so reshape then transpose to put leads first.
    signals = np.stack(df["signal"].values).reshape(len(df), 1000, 12)
    signals = np.ascontiguousarray(signals.transpose(0, 2, 1), dtype = np.float32)

    return signals, df["diagnostic_superclass"].map(LABEL_ENCODING)


def forward_fill(values):
    """Forward fill along the beat axis of one record."""
    if len(values) == 0:
        return values
    index = np.where(~np.isnan(values), np.arange(len(values)), 0)
    np.maximum.accumulate(index, out = index)
    return values[index]


def equate(values, length):
    """Truncate or pad one record to a common number of beats."""
    out = np.full(length, np.nan, dtype = np.float32)
    keep = min(len(values), length)
    if keep:
        out[:keep] = values[:keep]
        out[keep:] = values[keep - 1]     # forward fill past the last beat
    return out


def neurokit_features():
    """The per-lead matrix: 12 leads x 16 features x N_BEATS beats, flattened."""
    features = pd.read_parquet(DATA / "features_perlead.parquet")
    names = [c for c in features.columns if c != "label"]

    # One R-peak count per lead, so take the median across all of them.
    counts = np.concatenate([features[c].map(len).values
                             for c in names if c.endswith("_R_loc")])
    n_beats = int(np.median(counts))

    cube = np.stack([
        np.stack([equate(forward_fill(features[c].iloc[i]), n_beats) for c in names])
        for i in range(len(features))
    ])

    columns = [f"{c}_{k}" for c in names for k in range(n_beats)]
    prepared = pd.DataFrame(cube.reshape(len(cube), -1), index = features.index,
                            columns = columns).ffill()

    # Each lead's RR_0 is undefined, so forward fill has nothing to draw from.
    return prepared.drop(columns = prepared.columns[prepared.isna().all()])


def cnn_features(checkpoint, signals, batch_size):
    """The trained CNN's 125 features per record, extractor frozen."""
    extractor = CNNFeatureExtractor().to(DEVICE)
    extractor.load_state_dict(torch.load(checkpoint, map_location = DEVICE)["extractor"])
    extractor.eval()

    loader = DataLoader(TensorDataset(torch.from_numpy(signals)), batch_size = batch_size)
    with torch.no_grad():
        return np.concatenate([extractor(batch.to(DEVICE)).cpu().numpy()
                               for (batch,) in loader])


def training_loader(tensors, batch_size):
    """Shuffled loader over the 80 % split; the rest is never touched here."""
    train_index, _ = train_test_split(np.arange(len(tensors[0])),
                                      test_size = 0.2, random_state = SEED)
    return DataLoader(TensorDataset(*[t[train_index] for t in tensors]),
                      batch_size = batch_size, shuffle = True)


def fit(model, train, args, loss_of):
    """Adam loop shared by both models; loss_of supplies the difference.

    Returns the per-epoch mean training loss, saved with the checkpoint so the
    curve can be plotted without re-running the job.
    """
    optimiser = torch.optim.Adam(model.parameters(), lr = args.lr)
    loss_history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for batch in train:
            optimiser.zero_grad()
            loss = loss_of(model, batch)
            loss.backward()
            optimiser.step()
            total += loss.item() * len(batch[0])

        loss_history.append(total / len(train.dataset))
        print(f"  epoch {epoch:3d}  train loss {loss_history[-1]:.4f}")

    return loss_history



def train_cnn(args):
    signals, labels = load_signals()
    train = training_loader([torch.from_numpy(signals),
                             torch.from_numpy(labels.to_numpy(dtype = np.int64))],
                            args.batch_size)
    print(f"CNN on {len(train.dataset):,} training records"
          f"   ({len(signals) - len(train.dataset):,} held out)")

    model = CNNModel(num_classes = len(LABEL_ENCODING)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    loss_history = fit(model, train, args,
                       lambda m, b: criterion(m(b[0].to(DEVICE)), b[1].to(DEVICE)))

    torch.save({"model": model.state_dict(),
                "extractor": model.feature_extractor.state_dict(),
                "label_encoding": LABEL_ENCODING,
                "loss_history": loss_history}, args.out)
    print(f"saved {args.out}")


def train_ae(args):
    signals, labels = load_signals()
    prepared = neurokit_features()
    assert prepared.index.equals(labels.index), "feature and signal records differ"

    learned = cnn_features(args.cnn_checkpoint, signals, args.batch_size)
    matrix = np.hstack([prepared.to_numpy(dtype = np.float32), learned])
    print(f"autoencoder input: {prepared.shape[1]} neurokit"
          f" + {learned.shape[1]} CNN = {matrix.shape[1]}")

    # Fit on training rows only; the feature families differ by orders of
    # magnitude, so without this the loss is all sample-index error.
    train_index, _ = train_test_split(np.arange(len(matrix)), test_size = 0.2,
                                      random_state = SEED)
    scaler = StandardScaler().fit(matrix[train_index])
    scaled = torch.from_numpy(scaler.transform(matrix).astype(np.float32))

    train = training_loader([scaled], args.batch_size)
    model = Autoencoder(input_dim = matrix.shape[1], latent_dim = args.latent_dim).to(DEVICE)
    criterion = nn.MSELoss()

    def reconstruct(m, b):
        batch = b[0].to(DEVICE)
        return criterion(m(batch), batch)

    loss_history = fit(model, train, args, reconstruct)

    # Scaler kept as tensors rather than the sklearn object so the checkpoint
    # still loads under torch.load's weights_only default. To encode later:
    # model.encoder((x - scaler_mean) / scaler_scale)
    torch.save({"model": model.state_dict(),
                "encoder": model.encoder.state_dict(),
                "scaler_mean": torch.from_numpy(scaler.mean_),
                "scaler_scale": torch.from_numpy(scaler.scale_),
                "input_dim": matrix.shape[1],
                "latent_dim": args.latent_dim,
                "loss_history": loss_history}, args.out)
    print(f"saved {args.out}")



def main():
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--model", choices = ["cnn", "ae"], required = True)
    parser.add_argument("--epochs", type = int, default = 50)
    parser.add_argument("--batch-size", type = int, default = 64)
    parser.add_argument("--lr", type = float, default = 1e-3)
    parser.add_argument("--latent-dim", type = int, default = 16)
    parser.add_argument("--cnn-checkpoint", help = "required for --model ae")
    parser.add_argument("--out", required = True)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    print(f"device: {DEVICE}")

    if args.model == "cnn":
        train_cnn(args)
    else:
        if not args.cnn_checkpoint:
            parser.error("--model ae needs --cnn-checkpoint")
        train_ae(args)


if __name__ == "__main__":
    main()
