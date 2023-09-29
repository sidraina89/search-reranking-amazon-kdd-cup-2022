from data_factories import *
import pandas as pd
from typing import Union, List
import logging

features = [
    "query_id",
    "query",
    "query_locale",
    "product_id",
    "esci_label",
    "product_title",
]


class Process:
    def __init__(self, esciLoader, productLoader):
        self.esciLoader = esciLoader
        self.productLoader = productLoader

    def run(self, encode_labels: bool = None) -> pd.DataFrame:
        esci_data = self.esciLoader.get_search_ESCI_data()
        products = self.productLoader.get_products()
        esci_data_w_meta = pd.merge(
            esci_data.reset_index(drop=True),
            products.reset_index(drop=True),
            on="product_id",
            how="left"
        )
        if encode_labels:
            label_dict = {
                "exact": 1,
                "substitute": 0.1,
                "complement": 0.01,
                "irrelevant": 0,
            }
            esci_data_w_meta["esci_label"] = esci_data_w_meta["esci_label"].map(
                lambda x: label_dict[x]
            )
        return esci_data_w_meta[features]
