# Bank Marketing Decision Tree

This project trains a Decision Tree classifier using the UCI Bank Marketing dataset (the dataset is in a GitHub mirror provided in the prompt).

How to run (PowerShell):

```powershell
# create venv, install dependencies
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# run script
python task_project_3.py
```

What it does
- Downloads `bank-additional-full.csv`
- Preprocesses numeric and categorical features (standard scaling + one-hot)
- Trains a Decision Tree classifier and shows accuracy & classification report
- Outputs `decision_tree_bank_marketing.joblib` with the trained model
