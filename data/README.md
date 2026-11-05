# Data: UCI Banknote Authentication

| | |
|---|---|
| Source | Lohweg, V. (2012). *Banknote Authentication* [Dataset]. UCI Machine Learning Repository. <https://doi.org/10.24432/C55P57> |
| Licence | CC BY 4.0 |
| Download URL | <https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt> |
| Local file | `raw/banknote_authentication.csv` (byte-identical copy of the download) |
| SHA-256 | `d0539aaed2139ba7a587b3e34fb345ce503ff7d5d33dbf9912d8e195ce425cb9` |
| Origin paper | Lohweg et al. (2013), *Banknote authentication with mobile devices*, Proc. SPIE 8665. DOI 10.1117/12.2001444 |

The checksum is stored in `configs/default.yaml` and is checked on every run, so
a modified file stops the pipeline.

## Contents

No header; 1,372 rows; 5 comma-separated columns. The four features come from
a wavelet transform of a greyscale image of each banknote.

| Column | Name | Meaning |
|---|---|---|
| 1 | `variance` | variance of the wavelet-transformed image |
| 2 | `skewness` | skewness of the wavelet-transformed image |
| 3 | `curtosis` | kurtosis of the wavelet-transformed image (UCI spelling kept) |
| 4 | `entropy` | entropy of the image |
| 5 | `class` | 0 or 1 (762 / 610) |

The UCI page does not say which class value means "genuine". The code
therefore refers only to *class 0* and *class 1*. Labels become
y = −1 (class 0) and y = +1 (class 1).

## Verified facts

- 1,372 rows, no missing values, all features numeric, labels ⊂ {0, 1}
- 24 exact duplicate rows, all in class 0 → 1,348 unique rows (738 / 610)
- Only *exact* duplicates are removed. The dataset documentation does not say
  whether several images come from the same physical banknote, so near-duplicate
  leakage across splits cannot be ruled out.
- **Not linearly separable**, neither in ℝ⁴ nor on (variance, skewness): the
  hard-margin LP feasibility problem `y_i(<w, x_i> + b) >= 1` is infeasible. This is why
  the soft-margin / reduced-convex-hull theory is the relevant one.

## Preprocessing (in `src/svmdual/data.py`)

1. Verify the SHA-256 checksum and validate shape, types and labels.
2. Drop exact duplicate rows (keep first), so no identical note can appear in
   both training and test data.
3. Stratified 60 / 20 / 20 split into train / validation / test with the
   config seed: 808 / 270 / 270 rows at seed 42.
4. Standardise every feature with the **training** mean and standard deviation
   only, then apply that transform to validation and test.

The test split is used once, to evaluate the final models after (C, γ) have
been chosen on validation.
