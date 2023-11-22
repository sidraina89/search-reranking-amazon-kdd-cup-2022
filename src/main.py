from data_processor import Process
from eval import Evaluator
from model import CosineSimilarityModel
import logger
from data_downloader import download_data_from_hub

from data_factories import *
import yaml
from pathlib import Path
import logging,os

config_dict = yaml.safe_load(Path("../config/data_loc.yaml").read_text())
handler = logging.StreamHandler()
formatter = logger.OneLineExceptionFormatter(logging.BASIC_FORMAT)
handler.setFormatter(formatter)
root = logging.getLogger()
root.setLevel(os.environ.get("LOGLEVEL","INFO"))
root.addHandler(handler)

if __name__ == "__main__":
    # Download data to local
    download_data_from_hub()

    searchResultsLoader = ParquetSearchESCILoader(
        filePath=config_dict["local_data_dir"]
        + "/"
        + config_dict["data"]["train"]["filename"]
    )
    productLoader = ParquetProductCatalogLoader(
        filePath=config_dict["local_data_dir"]
        + "/"
        + config_dict["data"]["catalog"]["filename"]
    )

    train_df = Process(searchResultsLoader, productLoader).run(encode_labels=True)

    # Subset train_df to test on local
    # query_subset = train_df["query_id"].unique()[:1000]
    # train_df = train_df[train_df.query_id.isin(query_subset)]
    sim_model = CosineSimilarityModel()
    root.info("Embedding Queries..")
    train_df["query_embeddings"] = sim_model.forward(
        train_df["query"].values.tolist()
    ).tolist()
    root.info("Embedding Product Titles..")
    train_df["title_embeddings"] = sim_model.forward(
        train_df["product_title"].values.tolist()
    ).tolist()

    train_df["y_score"] = sim_model.predict(
        train_df.query_embeddings.values, train_df.title_embeddings.values
    )
    evaluator = Evaluator()
    root.info("ndcg@20: {}".format(evaluator.evaluate(train_df)))
