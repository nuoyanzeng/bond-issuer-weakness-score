# Sample CSV inputs

This directory contains five **synthetic** data records for each CSV input that is manually supplied to the main pipeline. The examples show filenames, column codes, and file layout; they are not research observations and cannot reproduce the report's results.

| Directory | Files | Layout |
| --- | --- | --- |
| [`china/`](china/) | Eight CSMAR exports and three Wind event exports | The CSMAR examples have a code header, a Chinese label row, a unit row, and five data rows. Their cells use CSMAR's `="value"` wrapping. The Wind examples have an English header, a Chinese label row, and five data rows. |
| [`selection/`](selection/) | Two manually prepared feature-selection CSVs | Each has a header and five example decisions. These are illustrative choices, not the complete selections required by the scripts. |

Export the actual CSMAR and Wind tables using authorized access and save them under `data_raw/china/` with the filenames shown here. Make your own feature-selection decisions and save the complete files under `output/`. See the root README for the required sources and selection rules. The `SignalDoc.csv` and CSMAR codebook metadata files are intentionally omitted; the root README explains how to obtain or build them.
