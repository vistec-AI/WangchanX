import numpy.random as random
import rich.repr
from datasets import Dataset, DatasetDict
from tqdm import tqdm


@rich.repr.auto
class DataGenerator:
    def __init__(self, patterns: dict):
        self.patterns = patterns

    def format_conversation(self, examples):
        if examples["input"]:
            examples["prompt"] = examples["instruction"] + "\n" + examples["input"]
        else:
            examples["prompt"] = examples["instruction"]
        examples["messages"] = [
            {"content": examples["prompt"], "role": "user"},
            {"content": examples["output"], "role": "assistant"},
        ]
        return dict(messages=examples["messages"], prompt=examples["prompt"])

    def _generate_patterned_data(self, dataset, pattern_name: str):
        if dataset is None:
            return
        patterns = random.choice(
            self.patterns[pattern_name], size=len(dataset), replace=True
        )
        for i, row in enumerate(dataset):
            pattern = patterns[i]
            data = {
                "instruction": pattern["instruction"].format(**row),
                "input": pattern["input"].format(**row),
                "output": pattern["output"].format(**row),
            }
            data = self.format_conversation(data)
            yield data

    def reformat(
        self,
        dataset: DatasetDict | Dataset,
        pattern_name: str | None = None,
        metadata: dict = {},
    ):
        if pattern_name is None:
            return dataset.map(lambda x: _add_metadata(x, **metadata))
        else:
            datas = []
            if isinstance(dataset, Dataset):
                generator = self._generate_patterned_data(dataset, pattern_name)
                datas.extend(generator)
            elif isinstance(dataset, DatasetDict):
                for key in ["train", "validation", "test"]:
                    dataset = dataset[key] if key in dataset else None
                    if dataset is not None:
                        generator = DataGenerator(
                            pattern_name
                        )._generate_patterned_data(dataset)
                        datas.extend(generator)
            else:
                raise TypeError("dataset_dict must be a Dataset or a DatasetDict")
            return Dataset.from_list(datas).map(lambda x: _add_metadata(x, **metadata))


def _add_metadata(examples, source: str, task: str, domain: str, license: str):
    examples["source"] = source
    examples["task"] = task
    examples["domain"] = domain
    examples["license"] = license
    return examples
