# A Fundamentals-Based Weakness Score for US and Chinese Bond Issuers

This project builds accounting-based measures of financial weakness for bond issuers. The models are trained on US issuers and applied to Chinese issuers without fitting on Chinese defaults or ratings. The resulting scores are intended as controls for issuer fundamentals in a broader study of relationships between bond issuers and their intermediaries.

For the research design, tables, figures, and interpretation, see the [report](A_Fundamentals_Based_Weakness_Score_for_US_and_Chinese_Bond_Issuers.pdf).

## Overview

The pipeline constructs 27 accounting characteristics, estimates three models (random forest, XGBoost, and elastic-net logistic regression), and produces two families of scores: one trained on US bond defaults or bankruptcy filings, and one trained on US speculative-grade ratings. US scores are generated out of fold; the models used to score Chinese issuers are fitted on the full US sample. The Chinese inputs are transformed to within-year ranks before scoring.

Selected findings from the report:

- Scores are available for 24,557 of 28,399 Chinese issuer-years (86.5%).
- In the US default sample, the highest out-of-fold area under the precision–recall curve is 0.145 for XGBoost, compared with a 1.22% event rate.
- In the 2013–2024 Chinese validation sample, default-prediction AUC-ROC ranges from 0.677 to 0.777 across the three transferred models. The report also examines financial outcomes and credit ratings.

These are research scores, not calibrated probabilities of default in China. See the report for sample definitions and limitations.

## Code organization

The scripts in [`code/`](code/) are numbered by stage:

| Stage | Purpose |
| --- | --- |
| 0–1 | Download US data, prepare accounting panels, and construct features |
| 2 | Clean features and assess redundancy |
| 3 | Link issuers, build labels and samples, and prepare model inputs |
| 4–5 | Evaluate US models, fit transfer models, and assemble US scores |
| 6 | Construct Chinese features and generate transferred scores |
| 7–8 | Compare ratings, validate scores, and produce exhibits |

Individual stage scripts document their expected inputs, outputs, and configuration near the top of each file.

## Data

Raw data and generated outputs are not included in this repository. Full reproduction requires access to the underlying sources and the exact extracts expected by the scripts.

| Source | Role | How it enters the project |
| --- | --- | --- |
| WRDS: Compustat, Mergent FISD, and CRSP identifier tables | US accounting data, bond events, ratings, and linking identifiers | With the appropriate WRDS access, set `WRDS_USERNAME` in [`code/stage0_download_wrds.py`](code/stage0_download_wrds.py) and run it from the repository root. It writes Parquet files to `data_raw/us/`. |
| [CSMAR](https://data.csmar.com/) | Chinese financial statements, bond and issuer information, and ratings | Export the required tables through an authorized CSMAR account and place the CSV files in `data_raw/china/`. The filenames and expected columns are specified in the Stage 6–8 scripts. |
| Wind Financial Terminal | Chinese bond defaults, extensions, and issuer/bond rating downgrades | Export the three event tables listed in [`code/stage8_2_default_labels.py`](code/stage8_2_default_labels.py) through an authorized Wind account and place the CSVs in `data_raw/china/` with the exact filenames shown there. |
| Chen–Zimmermann signal documentation | Characteristic definitions used by the feature inventory | Obtain the [version of `SignalDoc.csv` used in this project](https://github.com/OpenSourceAP/CrossSection/blob/8db892442c2c3a3779b0f1eac4370d3655be15a1/SignalDoc.csv) and place it at `data_raw/metadata/SignalDoc.csv`. |
| CSMAR dataset codebook | Optional coverage audit of the Chinese fields | Build `CSMAR_Dataset_Codebook.xlsx` from the field descriptions available through an authorized [CSMAR](https://data.csmar.com/) account. For each input CSV, add a section whose first cell starts with `文件` and names that table, followed by a header row (`英文变量名`, `对应中文名`, `变量说明`) and one row per field. Save it at `data_raw/metadata/CSMAR_Dataset_Codebook.xlsx`. Only `code/stage1_5_china_coverage.py` needs it. |

Neither metadata file is included in this repository. Supply the full files at the paths above when running the scripts that read them.

Five-row synthetic examples of the other manually supplied CSV inputs are in [`sample_data/`](sample_data/). They show the expected file layout but are not substitutes for the licensed source data or the complete feature selections.

## Running the code

Use Python 3 and run commands from the repository root. Dependencies are listed in [`requirements.txt`](requirements.txt).

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

After supplying the data, start the US extraction with:

```bash
python code/stage0_download_wrds.py
```

Continue through the numbered scripts in `code/`, checking each script's input and configuration notes before running it. Scripts write intermediate datasets, fitted models, score panels, and figures to `output/`.

### Manual feature selection

Feature selection is a research decision that users must make before running the later stages. Neither complete CSV is generated by the code or included in this repository; [`sample_data/selection/`](sample_data/selection/) shows only their format:

1. Before `stage1_3_merge_panels.py`, review the accounting characteristics and create `output/stage0_1_accounting_features.csv`. It must contain `Acronym` and `decision` columns; the script uses rows whose `decision` is exactly `keep` to build the feature inventory. The implemented feature panels and count checks reflect this project's original selection, so a different set also requires updating the construction code.
2. After reviewing the Stage 1.3 feature inventory, choose the features to retain and create `output/stage1_6_feature_selection.csv`. It must contain `feature` and `verdict` columns; `DROP` excludes a feature, and every other verdict keeps it. `stage2_1_clean_features.py` expects exactly 55 retained features.

**Reproduction status.** This is a research code release, not yet a one-command replication package. Full reproduction also requires access to the restricted raw data. A fresh clone cannot execute the complete pipeline end to end until the required data and the two manual selection files above are supplied. The report contains the results produced from the research environment; the code shows how the stages were implemented.

## References

The feature recipes build on Chen and Zimmermann (2022), *Open Source Cross-Sectional Asset Pricing*. The feature-screening approach also draws on Jensen, Kelly, and Pedersen (2023), *Is There a Replication Crisis in Finance?* Full references appear in the report.
