# Multi-Class Cardiovascular Disease Prediction from 12-Lead ECG

Reproduction of **DASMcC** on PTB-XL, followed by a proposed alternative pipeline.


---

## Components

| Path | Purpose |
|---|---|
| `src/data_loader.py` | Loads PTB-XL, aggregates SCP codes to the 5 diagnostic superclasses |
| `src/extract_nk_features.py` | `neurokit2` feature extraction cache|
| `notebooks/eda.ipynb` | Exploratory analysis; **builds `data/ptbxl100_single.parquet`** |
| `notebooks/reproduction.ipynb` | **Part 1** - faithful reproduction of DASMcC |
| `notebooks/ablation.ipynb` | **Part 1 ablation** — deliberate SMOTE-before-split leakage |
| `my_solution/CNN_extractor.py` | 1D CNN feature extractor (12×1000 → 125) |
| `my_solution/Autoencoder.py` | Autoencoder; its encoder compresses 2225 → 16 |
| `my_solution/train.py` | Trains the CNN and the autoencoder (80 % split only) |
| `my_solution/bash.sh` | Slurm submission script for the two training stages |
| `my_solution/my_notebook.ipynb` | **Part 2** — proposed pipeline results |
| `my_solution/ablation_no_encoder.ipynb` | **Part 2 ablation** — compressed vs. uncompressed features |
| `report/` | LaTeX source for the technical report |

---

## 1. Install

```bash
python3 -m venv .venv && source .venv/bin/activate    # Python 3.11
pip install -r requirements.txt
```

## 2. Get the data

PTB-XL v1.0.3 (~1.7 GB) from PhysioNet, manually extracted so that `records100/` and `ptbxl_database.csv` sit directly under `data/`.

## 3. Build the derived artefacts

Run in order. Each step writes into `data/` and is skipped if its output exists.

**Step 1: base table** (~5 min). Open `notebooks/eda.ipynb` and run all cells.

Uncomment this line on a first run, to produce `data/ptbxl100_single.parquet`:

```python
df_single.to_parquet(path + "ptbxl100_single.parquet")
```



**Step 2: neurokit2 features** (~20–40 min each, parallel over all cores):

```bash
cd src
python extract_nk_features.py timemajor   # Part 1  -> features_timemajor.parquet
python extract_nk_features.py perlead     # Part 2  -> features_perlead.parquet
python extract_nk_features.py leadmajor   # optional, for the flattening comparison
```


## 4. Reproduce the results

### Part 1: reproduction (CPU)

```bash
jupyter lab notebooks/reproduction.ipynb     # Run All
```

### Part 1 ablation: the leakage demonstration (CPU)

```bash
jupyter lab notebooks/ablation.ipynb         # Run All
```

Switch `FEATURES_PATH` between `features_timemajor.parquet` and
`features_leadmajor.parquet` to reproduce the flattening-order comparison.

### Part 2: training the extractors (GPU recommended)

Optional: checkpoints are supplied in `my_solution/checkpoints/`. To retrain, the two stages must be chained:

```bash
cd my_solution
cnn=$(sbatch --parsable --job-name=cnn bash.sh \
      --model cnn --epochs 500 --batch-size 64 --lr 1e-4 \
      --out checkpoints/cnn.pt)

sbatch --dependency=afterok:$cnn --job-name=ae16 bash.sh \
      --model ae --epochs 500 --batch-size 64 --lr 1e-4 --latent-dim 16 \
      --cnn-checkpoint checkpoints/cnn.pt --out checkpoints/ae16.pt
```

Without Slurm, call `train.py` directly with the same arguments.


### Part 2: results (CPU)

```bash
jupyter lab my_solution/my_notebook.ipynb            # Run All
jupyter lab my_solution/ablation_no_encoder.ipynb    # Run All
```

---

## Reproducibility notes

`random_state=42` throughout splits, SMOTE, every classifier, and `torch.manual_seed(42)` in `train.py`.