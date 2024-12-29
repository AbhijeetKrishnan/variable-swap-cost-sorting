# Variable Swap Cost Sorting

The accompanying code to my [blog post](http://abhijeetkrishnan.me/technical/variable-swap-cost-sort/).

## Reproduction

```bash
git clone git@github.com:AbhijeetKrishnan/variable-swap-cost-sorting.git
conda env create -p ./venv python pip polars matplotlib seaborn ipykernel
conda activate ./venv
python3 cost.py
```

This will produce `scores.csv` in the same directory, following which you can run all the cells in `analysis.ipynb`.