from .base_dataset import AbstractDataset
from datasets import load_dataset, load_from_disk, Dataset
import re
import rich.repr


@rich.repr.auto
class HuggingfaceDataset(AbstractDataset):
    @staticmethod
    def name(dataset, name) -> dict:
        dataset["name"] = name
        return dataset

    @staticmethod
    def source(dataset, source) -> dict:
        dataset["source"] = source
        return dataset

    @staticmethod
    def quantity(dataset) -> dict:
        if isinstance(dataset["datasets"], Dataset):
            dataset["original_quantity"] = len(dataset["datasets"])
        else:
            total_examples = sum(
                split.num_rows for split in dataset["datasets"].values()
            )
            dataset["original_quantity"] = total_examples
        return dataset

    @staticmethod
    def task(dataset, task) -> dict:
        dataset["task"] = task
        return dataset

    @staticmethod
    def domain(dataset, domain) -> dict:
        dataset["domain"] = domain

    @staticmethod
    def license(dataset, license) -> dict:
        dataset["license"] = license
        return dataset

    @staticmethod
    def download(dataset, cache_dir) -> dict:
        try:
            if dataset["source"].get("split"):
                datasets = load_from_disk(**dataset["source"])
                dataset["datasets"] = datasets
                return dataset
            datasets = load_dataset(
                **dataset["source"], split="train", cache_dir=cache_dir
            )
            dataset["datasets"] = datasets
            return dataset
        except:
            if dataset["source"].get("split"):
                datasets = load_dataset(**dataset["source"], cache_dir=cache_dir)
                dataset["datasets"] = datasets
                return dataset
            datasets = load_dataset(
                **dataset["source"], split="train", cache_dir=cache_dir
            )
            dataset["datasets"] = datasets
            return dataset

    @classmethod
    def download_dataset(
        cls,
        name,
        source,
        task,
        domain,
        license,
        processor,
        num_examples,
        file_format,
        template,
        cache_dir=None,
        **kwargs
    ) -> dict:
        _dataset = {}
        cls.name(_dataset, name)
        cls.source(_dataset, source)
        cls.download(_dataset, cache_dir)
        cls.quantity(_dataset)
        cls.task(_dataset, task)
        cls.domain(_dataset, domain)
        cls.license(_dataset, license)
        _dataset["template"] = template
        _dataset["processor"] = processor
        _dataset["num_examples"] = num_examples
        _dataset["file_format"] = file_format
        return _dataset