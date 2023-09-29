import pandas as pd
from typing import Union, List


class ParquetSearchESCILoader:
    def __init__(self, filePath: str):
        self.filePath = filePath

    def get_search_ESCI_data(self):
        return pd.read_parquet(self.filePath)


class ParquetProductCatalogLoader:
    def __init__(self, filePath: str):
        self.filePath = filePath

    def get_products(self):
        return pd.read_parquet(self.filePath)
