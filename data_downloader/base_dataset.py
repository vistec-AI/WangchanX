from abc import ABCMeta, abstractmethod
import os
from urllib.parse import urlparse
from datasets import load_dataset


def is_url(path):
    try:
        result = urlparse(path)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def extract_filename(base_filename, extensions):
    if is_url(base_filename):
        base_filename = base_filename.split("/")[-1]
    name_parts = base_filename.split(".")
    if len(name_parts) > 1 and name_parts[-1].lower() in extensions:
        filename = ".".join(name_parts[:-1])
        return filename
    else:
        return None


def load_file_datasets(path, file_format, **kwargs):
    if is_url(path):
        dataset = load_dataset(file_format, data_files=path, **kwargs)
    else:
        file_extensions = {
            "csv": [".csv"],
            "tsv": [".tsv", ".tab"],
            "json": [".json", ".jsonl"],
        }
        extensions = file_extensions.get(file_format, [])

        data_files = [
            os.path.join(path, file)
            for file in os.listdir(path)
            if any(file.endswith(ext) for ext in extensions)
        ]

        dataset = load_dataset(file_format, data_files=data_files, **kwargs)

    return dataset


class AbstractDataset(metaclass=ABCMeta):

    @staticmethod
    @abstractmethod
    def name(dataset):
        pass

    @staticmethod
    @abstractmethod
    def source(dataset):
        pass

    @staticmethod
    @abstractmethod
    def quantity(dataset):
        pass

    @staticmethod
    @abstractmethod
    def task(dataset):
        pass

    @staticmethod
    @abstractmethod
    def domain(dataset):
        pass

    @staticmethod
    @abstractmethod
    def license(dataset):
        pass

    @staticmethod
    @abstractmethod
    def download(dataset):
        pass

    @classmethod
    def download_dataset(cls):
        _dataset = {}
        cls.name(_dataset)
        cls.source(_dataset)
        cls.download(_dataset)
        cls.quantity(_dataset)
        cls.task(_dataset)
        cls.domain(_dataset)
        cls.license(_dataset)
