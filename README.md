# ChurnEDA

Unsupervised churn exploration and visualization using Python. This project loads training and test CSV files, performs basic cleaning and standardization, and generates a set of exploratory plots for both univariate and multivariate analysis to better understand churn-related patterns.

## Project Structure

```
ChurnEDA/
├── data/
│   ├── train.csv
│   └── test.csv
├── main.py
├── utils.py
├── requirements.txt
└── README.md
```

- main.py: Orchestrates data loading, preprocessing, and calls plotting utilities.
- utils.py: Collection of plotting helper functions (bar plots, histograms, boxplots, heatmaps, stacked bars, grouped plots, scatter matrix).
- data/: Contains the CSV files used by the analysis.

## Requirements

See requirements.txt. Key libraries:
- pandas
- numpy
- matplotlib
- seaborn

## Installation

1. Create and activate a virtual environment (recommended):
   - Python 3.10+ is recommended.
   - On macOS/Linux:
     - python3 -m venv .venv
     - source .venv/bin/activate
   - On Windows (PowerShell):
     - py -3 -m venv .venv
     - .venv\Scripts\Activate.ps1

2. Install dependencies:

```
pip install -r requirements.txt
```

## Data

Place your CSVs under the data/ directory with the following expected files:
- data/train.csv
- data/test.csv

Expected columns include at least:
- customerID: identifier (will be dropped)
- churn: integer flag where 1 denotes churn and 0 denotes no churn (converted to categorical)

Other columns can be numerical or categorical. Column names are standardized to lowercase with underscores. String values are lowercased and spaces replaced with underscores.

## Usage

Run the analysis script from the project root:

```
python main.py
```

The script will:
- Load train and test CSVs and concatenate them.
- Clean columns and types.
- Print heads, dtypes, and missing value summaries.
- Generate plots:
  - Univariate: bar plots for categorical variables; histograms and boxplots for numeric variables.
  - Multivariate: stacked bar plots, grouped box/violin plots, correlation heatmap, and a scatterplot matrix.

Plots are shown interactively using matplotlib/seaborn. If running in a headless environment, you may need to set a non-interactive backend or save figures instead of showing them.

## Notes

- Ensure your CSV headers match the expected fields. The script drops a column named customerid (case-insensitive after standardization) and expects a churn column coded as 0/1.
- The code uses plt.style.use("fivethirtyeight") for consistent visuals.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Author

- Implementation: Ramiro Saltos
- Date: 2025-08-30
