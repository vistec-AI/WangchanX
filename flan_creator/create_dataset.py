import random
from random import Random
from functools import partial
from typing import Any, List, Set, Tuple, Union

import pandas as pd
from datasets import Dataset, DatasetDict, concatenate_datasets
from numpy import concatenate
from parallel_pandas import ParallelPandas
from rich.pretty import pprint
from tabulate import tabulate

from data_downloader import *
from data_formatter.formatter import DataGenerator
from data_processor import *


class DatasetManager:
    def __init__(
        self,
        templates: dict,
        dataset_metadata: dict,
        cache_dir: str = None,
        save_dir: str = "flan_datasets",
        save_format: str = "arrow",
        save_metadata: str = "metadata.txt",
        num_proc: int = 1,
        drop_mode: str = "huggingface",
    ):
        self._dataset_metadata = dataset_metadata
        self._datasets = None
        self._cache_dir = cache_dir
        self._save_dir = save_dir
        if save_format not in ["csv", "arrow", "jsonl"]:
            raise ValueError("save_format must be one of ['csv', 'arrow', 'jsonl']")
        self._save_format = save_format
        self._save_metadata = save_metadata
        self._data_generator = DataGenerator(templates)
        self._num_proc = num_proc
        self._drop_mode = drop_mode
        if self._drop_mode:
            self._drop_duplicates = True
        else:
            self._drop_duplicates = False
        ParallelPandas.initialize(n_cpu=num_proc, split_factor=4, disable_pr_bar=False)

    def _get_downloader(self, file_format: str) -> callable:
        """Return a downloader based on the file format."""
        downloaders = {
            "csv": CSVDataset.download_dataset,
            "huggingface": HuggingfaceDataset.download_dataset,
            "json": JsonDataset.download_dataset,
        }
        return downloaders.get(file_format)

    def _download(self, name: str) -> None:
        """Download a dataset and store the info."""
        assert name in self._dataset_metadata, f"{name} not found in dataset_info.json"
        info = self._dataset_metadata[name]
        file_format = info["file_format"]
        downloader = self._get_downloader(file_format)
        if downloader:
            self._dataset_metadata[name] = downloader(
                name, **info, cache_dir=self._cache_dir
            )
        else:
            raise ValueError("file_format must be csv/tsv, huggingface or json")

    def _get_processor(self, name) -> callable:
        """Return a processor based on the dataset info."""
        processor_class = self._dataset_metadata[name].get("processor")
        if processor_class:
            return eval(processor_class)
        else:
            raise ValueError("Processor not found in dataset_info")

    def get_cleaner(self, name: str) -> callable:
        """Return a cleaner based on the dataset info."""
        processor = self._get_processor(name)
        return processor(
            self._dataset_metadata[name]["datasets"], num_proc=self._num_proc
        )

    def clean(self, name: str):
        """Clean a dataset."""
        cleaner = self.get_cleaner(name)
        num_examples = self._dataset_metadata[name].get("num_examples")
        dataset = cleaner.deterministic_random(num_examples)
        self._dataset_metadata[name]["datasets"] = dataset.clean()

    def reformat(self, name):
        """Reformat a dataset."""
        template = self._dataset_metadata[name]["template"]
        datasets = self._dataset_metadata[name]["datasets"]
        source = self._dataset_metadata[name]["source"]["path"]
        task = self._dataset_metadata[name]["task"]
        license = self._dataset_metadata[name]["license"]
        license_str = ", ".join(license) if type(license) == list else license
        domain = self._dataset_metadata[name]["domain"]
        metadata = {
            "source": source,
            "task": task,
            "license": license_str,
            "domain": domain,
        }
        reformatted_dataset = self._data_generator.reformat(
            datasets,
            template,
            metadata=metadata,
        )
        self._dataset_metadata[name]["datasets"] = reformatted_dataset

    def concat_dataset(self):
        return concatenate_datasets(
            [
                self._dataset_metadata[name]["datasets"]
                for name in self._dataset_metadata
            ]
        )

    def _is_unique(self, elem: Any, columns: List[str], memory: Set[Tuple]) -> bool:
        key = tuple(str(elem[column]) for column in columns)
        if key in memory:
            return False
        else:
            memory.add(key)
            return True

    def drop_duplicates(self):
        datasets = self.concat_dataset()
        if self._drop_mode == "huggingface":
            memory = set()
            self._datasets = datasets.filter(
                partial(self._is_unique, columns=["messages"], memory=memory),
                desc="Remove duplicates",
                num_proc=self._num_proc,
            )
        elif self._drop_mode == "pandas":
            datasets.set_format(type="pandas")
            datasets = datasets[:]
            datasets["text"] = datasets["messages"].apply(lambda x: str(x))
            datasets = datasets.drop_duplicates(subset=["text"], keep="first")
            self._datasets = Dataset.from_pandas(datasets).select_columns(
                ["messages", "prompt"]
            )
        else:
            raise ValueError("mode must be huggingface or pandas")

    def save_metadata(self):
        info = self._dataset_metadata.copy()
        for name in info:
            info[name]["processed_quantity"] = len(info[name]["datasets"])
        metadata_df = pd.DataFrame.from_dict(info, orient="index")
        metadata_df = metadata_df[["name", "original_quantity", "processed_quantity"]]
        metadata_df.columns = ["name", "original_quantity", "processed_quantity"]
        total_o = metadata_df["original_quantity"].sum()
        total_p = metadata_df["processed_quantity"].sum()
        metadata_df["name"] = metadata_df["name"].apply(
            lambda x: x if isinstance(x, str) else x[0]
        )
        max_string = metadata_df["name"].str.len().max()
        half_string_max = max_string // 2
        half_string_max = "=" * half_string_max
        string_total = half_string_max + " total " + half_string_max
        metadata_df = metadata_df._append(
            {
                "name": string_total,
                "original_quantity": total_o,
                "processed_quantity": total_p,
            },
            ignore_index=True,
        )

        with open(self._save_metadata, "w") as f:
            f.write(tabulate(metadata_df, headers="keys", tablefmt="psql"))
        return metadata_df

    def save_datasets(self) -> None:
        """Save the datasets."""
        self._datasets = self.concat_dataset()
        self._datasets = DatasetDict({"train": self._datasets})
        if self._save_format == "json" or self._save_format == "jsonl":
            self._datasets.to_json(self._save_dir)
        elif self._save_format == "csv":
            self._datasets.to_csv(self._save_dir)
        elif self._save_format == "arrow":
            self._datasets.save_to_disk(self._save_dir)
        else:
            raise ValueError("save_format must be one of ['json', 'csv', 'arrow']")

    def get_examples(self, name):
        dataset = self._dataset_metadata[name]["datasets"]
        random_ids = Random(42).sample(range(len(dataset)), 2)
        return dataset.select(random_ids)["messages"]
