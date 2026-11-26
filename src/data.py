from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def load_destinations():
    return pd.read_csv(ROOT/"data"/"destinations.csv")
def load_attractions():
    return pd.read_csv(ROOT/"data"/"attractions.csv")
