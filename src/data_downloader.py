from logger import OneLineExceptionFormatter

import requests, yaml
from pathlib import Path
from huggingface_hub import hf_hub_download
from tqdm import tqdm
import logging

log = logging.getLogger(__name__)

config_dict = yaml.safe_load(Path("../config/data_loc.yaml").read_text())

def download_data_from_hub():
    log.info("Downloading data files from HF Hub ..")
    for file_type in tqdm(config_dict["data"].keys()):
        hf_hub_download(repo_id=config_dict["hf_hub_repo"],
                        filename=config_dict["data"][file_type]["filename"],
                        local_dir=config_dict["local_data_dir"],
                        local_dir_use_symlinks=False,
                        repo_type="dataset")