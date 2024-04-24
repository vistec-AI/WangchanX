from abc import ABC, abstractmethod
import numpy as np
from typing import List, Any, Set, Tuple, Union, Optional
from datasets import Dataset, DatasetDict
from functools import partial
import emoji
import random as random_normal
import re
from tqdm import tqdm


class CleaningStrategy(ABC):
    def set_num_proc(self, num_proc: int):
        self.num_proc = num_proc

    def dropna(self, columns: List[str]):
        for column in columns:
            self.dataset = self.dataset.filter(
                lambda x: x[column] is not None,
                desc=f"Remove null values column {column}",
                num_proc=self.num_proc,
            )
        return self

    def deterministic_random(self, k: Union[str, int] = "all", seed: int = 42):

        total = len(self.dataset)

        assert k in ["all", "half", "quarter"] or (
            isinstance(k, int) and 0 < k <= total
        ), "k should be ['all', 'half', 'quarter'] or a positive integer less than or equal to the total examples of the dataset"

        if k == "all":
            random_ids = random_normal.Random(seed).sample(
                range(len(self.dataset)), total
            )
        elif k == "half":
            random_ids = random_normal.Random(seed).sample(
                range(len(self.dataset)), int(np.ceil(total / 2))
            )
        elif k == "quarter":
            random_ids = random_normal.Random(seed).sample(
                range(len(self.dataset)), int(np.ceil(total / 4))
            )
        else:
            random_ids = random_normal.Random(seed).sample(range(total), k)

        self.dataset = self.dataset.select(random_ids)
        return self

    def _is_unique(self, elem: Any, columns: List[str], memory: Set[Tuple]) -> bool:
        key = tuple(str(elem[column]) for column in columns)
        if key in memory:
            return False
        else:
            memory.add(key)
            return True

    def drop_duplicates(
        self,
        columns: List[str],
        num_proc: int = 1,
    ) -> Dataset:

        memory = set()
        dataset = self.dataset.filter(
            partial(self._is_unique, columns=columns, memory=memory),
            desc="Remove duplicates",
            num_proc=self.num_proc,
        )
        return dataset

    @abstractmethod
    def clean(self, dataset):
        pass


class SentimentStrategy(CleaningStrategy):
    def __init__(self, dataset: Dataset, feelings: List[dict], num_proc: int = 1):
        assert isinstance(dataset, Dataset), "dataset must be an instance of Dataset"
        self.dataset = dataset
        self.feeling_templates = [str(feeling) for feeling in feelings]
        self.dataset = self.dataset.add_column(
            "feelings_mapping", self.random_feeling()
        )
        self.set_num_proc(num_proc)

    @abstractmethod
    def normalize_answer(self, example):
        pass

    @abstractmethod
    def extract_feelings(self, example):
        pass

    def random_feeling(self):
        return np.random.choice(
            self.feeling_templates, size=len(self.dataset), replace=True
        )

    @abstractmethod
    def clean(self, dataset):
        pass


class ColumnStrategy(CleaningStrategy):
    def __init__(self, dataset: Dataset, num_proc: int = 1):
        assert isinstance(dataset, Dataset), "dataset must be an instance of Dataset"
        self.dataset = dataset
        self.set_num_proc(num_proc)

    def replace_column(self, dataset: Dataset, source: str, target: str):
        dataset = dataset.remove_columns([source])
        dataset = dataset.rename_column(target, source)
        return dataset

    def translate_columns(self, example, lang_cols):
        for col in lang_cols:
            example[col] = example["translation"][col]
        return example

    @abstractmethod
    def clean(self, dataset):
        pass


class TextStrategy(CleaningStrategy):
    def __init__(self, dataset: Dataset, num_proc: int = 1):
        assert isinstance(dataset, Dataset), "dataset must be an instance of Dataset"
        self.dataset = dataset
        self.set_num_proc(num_proc)

    def remove_emoji(self, example):
        example["text"] = emoji.replace_emoji(example["text"], replace="")
        return example

    def add_prefix(self, example):
        a = example["text"].split("<bot>:")

        example["bot_morethan_one"] = len(a)
        example["has_context"] = 1 if "<context>:" in example["text"] else 0

        v = example["text"]

        context_index = v.find("<context>:")
        human_index = v.find("<human>:")
        bot_index = v.find("<bot>:")

        context = v[context_index:human_index].replace("<context>:", "").strip()
        human = v[human_index:bot_index].replace("<human>:", "").strip()
        bot = v[bot_index:].replace("<bot>:", "").strip()

        combined = ""
        if context != "":
            combined = context + "\n" + human
        else:
            combined = human

        example["Context"] = ""
        example["Instruction"] = combined.strip()
        example["Answer"] = bot.strip()

        return example

    def extract_dialogue_components(self, example):
        pattern = r"<human>:\s*(.*?)\s*<context>:\s*(.*?)\s*<bot>:\s*(.*)"
        extract = re.match(pattern, example["text"])
        human, context, bot = extract.groups()
        example["human"] = human
        example["context"] = context
        example["bot"] = bot
        return example

    def replace_language_codes_with_names(self, example, language_map):
        for lang_key, lang_values in language_map.items():
            if example["source_language"] == lang_key:
                example["source_lang"] = random_normal.choice(lang_values)
            if example["target_language"] == lang_key:
                example["target_lang"] = random_normal.choice(lang_values)
        return example

    @abstractmethod
    def clean(self, dataset):
        pass
