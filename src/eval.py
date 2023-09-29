from sklearn.metrics import ndcg_score
from typing import Dict
import pandas as pd
import numpy as np


class Evaluator:
    def __init__(self):
        pass

    def evaluate(self,df: pd.DataFrame, k=20):
        grouped_df = df.groupby("query_id").aggregate(
            {
                "y_score": lambda x: list(x),
                "esci_label": lambda x: list(x),
            }
        ).reset_index()
        grouped_df["ndcg_score"] = grouped_df.apply(
            lambda x: ndcg_score(y_true=[x["esci_label"]], y_score=[x["y_score"]], k=20), axis=1
        )
        return grouped_df["ndcg_score"].mean()
