import os
from dotenv import load_dotenv
import torch
import numpy as np
from openai import AzureOpenAI
import logging
from typing import Any, Union
from abc import ABC
from mteb import MTEB
import json
import pandas as pd

load_dotenv()

def get_logger() -> logging.Logger:
    """Returns a configured logger"""
    logging.basicConfig(
        filename="mteb.log",
        filemode="a",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    return logging.getLogger("main")


logger = get_logger()

class ClientManager:
    def __init__(self):
        self.oai_client = None

    def get_oai_client(self):
        if self.oai_client is None:
            self.oai_client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_GPT")
            )
        return self.oai_client

class ExternalModel(ABC):
    """Abstract class for models accessible via external API"""

    def __init__(self, model_name: str, client: Any, **kwargs):
        self.model_name = model_name
        self.client = client

    def encode(
        self, sentences: list[str], batch_size: int, **kwargs: Any
    ) -> Union[torch.Tensor, np.ndarray]:
        """Encodes the given sentences using the encoder.

        Args:
            sentences: The sentences to encode.
            batch_size (`int`): Batch size for the encoding
            **kwargs: Additional arguments to pass to the encoder.

        Returns:
            The encoded sentences.
        """
        pass

class OpenAIAPIModel(ExternalModel):
    def encode(self, sentences: list[str], **kwargs: Any) -> np.ndarray:
        logger.info(f"Encoding {len(sentences)} sentences")
        embeddings = []
        for sentence in sentences:
            try:
                response = self.client.embeddings.create(input=sentence, model=self.model_name)
                embeddings.append(response.data[0].embedding)
            except Exception as e:
                logger.error(f"Error encoding sentence: {e}")
                raise
        return np.array(embeddings)

def run_law_benchmarks(model: ExternalModel, logger: logging.Logger):
    """Runs the law benchmarks for the given model via API"""

    model_name = model.model_name

    TASK_LIST_RETRIEVAL_LAW = {
        "LegalSummarization": "corpus",  # Specify the configuration
        "LegalBenchConsumerContractsQA": "default",  # Specify the configuration
        "LegalBenchCorporateLobbying": "default",  # Specify the configuration
    }

    for task, config in TASK_LIST_RETRIEVAL_LAW.items():
        logger.info(f"Running task: {task} with configuration: {config} for model: {model_name}")
        eval_splits = ["test"]
        evaluation = MTEB(tasks=[task], config=config)  # Specify the configuration here
        evaluation.run(
            model, output_folder=f"results/{model_name}", eval_splits=eval_splits
        )

def get_partial_results_json(path: str) -> dict:
    """Returns a dictionary with the results of the vanilla benchmarks"""
    with open(path, "r") as f:
        results = json.load(f)
    return {
        "mteb_dataset_name": results["mteb_dataset_name"],
        "evaluation_time": results["test"]["evaluation_time"],
        "ndcg@10": results["test"]["ndcg_at_10"],
    }

def get_cross_lingual_results_json(path: str) -> dict:
    """Returns a dictionary with the results of the cross-lingual benchmarks"""
    with open(path, "r") as f:
        results = json.load(f)
    scores = extract_bitext_scores(results["test"])
    return {
        "mteb_dataset_name": results["mteb_dataset_name"],
        "evaluation_time": results["test"]["evaluation_time"],
        "f1": scores["f1"],
    }


def get_all_results():
    """Returns a both dictionaries with the results of the vanilla and cross-lingual benchmarks"""

    all_results = []
    cross_lingual_results = []
    for model_name in os.listdir("results/"):
        path = f"results/{model_name}"
        if os.path.isdir(path):
            for file in os.listdir(path):
                file_path = f"{path}/{file}"
                if not file.startswith("Tatoeba"):
                    d = get_partial_results_json(file_path)
                    d["model_name"] = model_name
                    all_results.append(d)
                else:
                    d = get_cross_lingual_results_json(file_path)
                    d["model_name"] = model_name
                    cross_lingual_results.append(d)
    return all_results, cross_lingual_results

def get_vanilla_results_df(values=["ndcg@10", "evaluation_time"]) -> pd.DataFrame:
    """Returns a DataFrame with the vanilla results"""

    df = pd.DataFrame(get_all_results()[0])
    pivot_df = df.pivot_table(
        columns="mteb_dataset_name",
        index="model_name",
        values=values,
        aggfunc="mean",
        margins=True,
        margins_name="Average",
    )
    return pivot_df


def get_cross_lingual_results_df(values=["scores", "evaluation_time"]) -> pd.DataFrame:
    """Returns a DataFrame with the cross-lingual results"""

    df = pd.DataFrame(get_all_results()[1])
    pivot_df = df.pivot_table(
        columns="mteb_dataset_name",
        index="model_name",
        values=values,
        aggfunc="mean",
        margins=True,
        margins_name="Average",
    )
    return pivot_df