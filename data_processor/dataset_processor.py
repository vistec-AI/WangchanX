from typing import List
from datasets import Dataset
import numpy.random as random
from .base_data_processor import ColumnStrategy, TextStrategy, SentimentStrategy
import rich.repr

SENTIMENT_MAPPINGS = [
    {2: "เชิงลบ", 1: "เป็นกลาง", 0: "เชิงบวก"},
    {2: "negative", 1: "neutral", 0: "positive"},
    {2: "รู้สึกแย่", 1: "รู้สึกเฉยๆ", 0: "รู้สึกดี"},
    {2: "bad", 1: "neutral", 0: "good"},
    {2: "terrible", 1: "neutral", 0: "great"},
]


@rich.repr.auto
class Xp3x(TextStrategy):
    def clean(self):
        dataset = self.dropna(columns=["inputs"]).drop_duplicates(columns=["inputs"])
        return dataset


@rich.repr.auto
class WongnaiSentiment(SentimentStrategy, ColumnStrategy):
    def __init__(
        self,
        dataset: Dataset,
        feelings: List[dict] = SENTIMENT_MAPPINGS,
        num_proc: int = 1,
    ):
        super().__init__(dataset, feelings, num_proc)

    def normalize_answer(self, example):
        def convert_star_rating_to_sentiment(rating):
            if rating == 0:
                return eval(example["feelings_mapping"])[2]
            elif rating == 1:
                return eval(example["feelings_mapping"])[2]
            elif rating == 2:
                return eval(example["feelings_mapping"])[2]
            elif rating == 3:
                return eval(example["feelings_mapping"])[1]
            elif rating == 4:
                return eval(example["feelings_mapping"])[0]
            elif rating == 5:
                return eval(example["feelings_mapping"])[0]
            else:
                raise ValueError(f"Invalid rating: {rating}")

        example["sentiment"] = convert_star_rating_to_sentiment(example["star_rating"])
        return example

    def extract_feelings(self, example):
        example["neu"] = eval(example["feelings_mapping"])[1]
        example["pos"] = eval(example["feelings_mapping"])[0]
        example["neg"] = eval(example["feelings_mapping"])[2]
        return example

    def clean(self):
        dataset = (
            self.dropna(columns=["review_body"])
            .drop_duplicates(columns=["review_body"])
            .map(self.normalize_answer, desc="Normalize answer", num_proc=self.num_proc)
            .map(self.extract_feelings, desc="Extract feelings", num_proc=self.num_proc)
        )
        return dataset


@rich.repr.auto
class WiseSightSentiment(SentimentStrategy, ColumnStrategy):
    def __init__(
        self,
        dataset: Dataset,
        feelings: List[dict] = SENTIMENT_MAPPINGS,
        num_proc: int = 1,
    ):
        super().__init__(dataset, feelings, num_proc)

    def normalize_answer(self, example):
        if example["category"] == 2:
            example["sentiment"] = eval(example["feelings_mapping"])[2]
        elif example["category"] == 1:
            example["sentiment"] = eval(example["feelings_mapping"])[1]
        elif example["category"] == 0:
            example["sentiment"] = eval(example["feelings_mapping"])[0]
        else:
            raise ValueError(f"Invalid answer: {example['category']}")
        return example

    def extract_feelings(self, example):
        example["answer"] = eval(example["feelings_mapping"])[example["category"]]
        example["neu"] = eval(example["feelings_mapping"])[1]
        example["pos"] = eval(example["feelings_mapping"])[0]
        example["neg"] = eval(example["feelings_mapping"])[2]
        return example

    def clean(self):
        dataset = (
            self.dropna(columns=["texts"])
            .drop_duplicates(columns=["texts"])
            .filter(lambda x: x["category"] in [0, 1, 2])
            .map(self.normalize_answer, desc="Normalize answer", num_proc=self.num_proc)
            .map(self.extract_feelings, desc="Extract feelings", num_proc=self.num_proc)
        )
        return dataset


@rich.repr.auto
class WiKiLingual(TextStrategy):
    def clean(self):
        thais = ["ไทย", "Thai"]
        englishs = ["อังกฤษ", "English"]
        dataset = (
            self.dropna(columns=["source"])
            .drop_duplicates(columns=["target"])
            .map(
                self.replace_language_codes_with_names,
                fn_kwargs={
                    "language_map": {"th": ["ไทย", "Thai"], "en": ["อังกฤษ", "English"]}
                },
                num_proc=self.num_proc,
            )
        )
        return dataset


@rich.repr.auto
class WangchanGLM(TextStrategy):
    def clean(self):
        dataset = (
            self.dropna(columns=["text"])
            .drop_duplicates(columns=["text"])
            .map(self.add_prefix, desc="Add prefix")
            .filter(
                lambda x: x["bot_morethan_one"] == 2,
                desc="Filter more than one bot",
                num_proc=self.num_proc,
            )
        )
        return dataset


@rich.repr.auto
class Ultrachat200k(TextStrategy):
    def clean(self):
        dataset = (
            self.dropna(columns=["messages"])
            .drop_duplicates(columns=["messages"])
            .select_columns(["messages", "prompt"])
        )
        return dataset


@rich.repr.auto
class TinyCodes(TextStrategy):
    def clean(self):
        dataset = (
            self.dropna(columns=["prompt"])
            .drop_duplicates(columns=["prompt"])
            .filter(
                lambda example: example["programming_language"]
                in ["JavaScript", "Python", "relation database and SQL"],
                num_proc=self.num_proc,
            )
        )
        return dataset


@rich.repr.auto
class ThaiSum(TextStrategy):
    def clean(self):
        dataset = self.dropna(columns=["body"]).drop_duplicates(columns=["body"])
        return dataset


@rich.repr.auto
class ThaiWiki(TextStrategy):
    def clean(self):
        dataset = self.dropna(columns=["text"]).drop_duplicates(columns=["text"])
        return dataset


@rich.repr.auto
class ThaiUSEmbassy(TextStrategy):
    def clean(self):
        dataset = self.dropna(columns=["th"]).drop_duplicates(columns=["th"])
        return dataset


@rich.repr.auto
class ThaiSentimentAnalysis(SentimentStrategy, TextStrategy):
    def __init__(self, dataset: Dataset, feelings: List[dict] = SENTIMENT_MAPPINGS, num_proc: int = 1):
        super().__init__(dataset, feelings, num_proc)

    def normalize_answer(self, example):
        if example["answer"] == "neg":
            example["sentiment"] = eval(example["feelings_mapping"])[2]
        elif example["answer"] == "neu":
            example["sentiment"] = eval(example["feelings_mapping"])[1]
        elif example["answer"] == "pos":
            example["sentiment"] = eval(example["feelings_mapping"])[0]
        else:
            raise ValueError(f"Invalid answer: {example['answer']}")
        return example

    def extract_feelings(self, example):
        example["neu"] = eval(example["feelings_mapping"])[1]
        example["pos"] = eval(example["feelings_mapping"])[0]
        example["neg"] = eval(example["feelings_mapping"])[2]
        return example

    def clean(self):
        dataset = self.dropna(columns=["answer"]).drop_duplicates(columns=["text"])
        dataset = (
            dataset.map(self.normalize_answer, desc="Normalize answer")
            .map(self.remove_emoji, desc="Remove emoji", num_proc=self.num_proc)
            .map(self.extract_feelings, desc="Extract feelings", num_proc=self.num_proc)
        )
        return dataset


@rich.repr.auto
class ThaiFood(TextStrategy):
    def clean(self):
        dataset = self.dropna(columns=["text"]).drop_duplicates(columns=["text"])
        return dataset


@rich.repr.auto
class ThaiEnglishTransliterationDictionary(TextStrategy):
    def clean(self):
        dataset = self.dropna(columns=["th"]).drop_duplicates(columns=["th"])
        return dataset


@rich.repr.auto
class TedTalks(ColumnStrategy):
    def clean(self):
        dataset = (
            self.dropna(columns=["translation"])
            .drop_duplicates(columns=["translation"])
            .map(
                self.translate_columns,
                desc="Translate columns",
                fn_kwargs={"lang_cols": ["en", "th"]},
                num_proc=self.num_proc,
            )
        )
        return dataset


@rich.repr.auto
class SCBEnTh(ColumnStrategy):
    def clean(self):
        dataset = (
            self.dropna(columns=["translation"])
            .drop_duplicates(columns=["translation"])
            .map(
                self.translate_columns,
                desc="Translate columns",
                fn_kwargs={"lang_cols": ["en", "th"]},
                num_proc=self.num_proc,
            )
        )
        return dataset


@rich.repr.auto
class PRDNews(TextStrategy):
    def clean(self):
        dataset = self.dropna(columns=["Detail"]).drop_duplicates(columns=["Detail"])
        return dataset


@rich.repr.auto
class Platypus(TextStrategy):
    def clean(self):
        dataset = (
            self.dropna(columns=["output"])
            .drop_duplicates(columns=["output"])
            .filter(
                lambda example: example["data_source"] != "scienceqa"
                and example["data_source"] != "reclor",
                desc="Filter scienceqa and reclor",
                num_proc=self.num_proc,
            )
        )
        return dataset


@rich.repr.auto
class Math50k(ColumnStrategy):
    def clean(self):
        dataset = self.dropna(columns=["output"]).drop_duplicates(
            columns=["instruction"]
        )
        return dataset


@rich.repr.auto
class KlongKlon(TextStrategy):
    def clean(self):
        dataset = (
            self.dropna(columns=["text"])
            .drop_duplicates(columns=["text"])
            .map(
                self.extract_dialogue_components,
                desc="Add prefix",
                num_proc=self.num_proc,
            )
        )
        return dataset


@rich.repr.auto
class IApp(ColumnStrategy):
    def clean(self):
        dataset = self.dropna(columns=["question", "answers"]).drop_duplicates(
            columns=["question"]
        )
        dataset = self.replace_column(dataset, "answers", "answers_text")
        return dataset


@rich.repr.auto
class Han(TextStrategy):
    def clean(self):
        dataset = self.dropna(columns=["q", "a"]).drop_duplicates(columns=["q", "a"])
        return dataset


@rich.repr.auto
class FlanV2(TextStrategy):
    def clean(self, k: int = 100000):
        dataset = self.dropna(columns=["inputs"]).drop_duplicates(columns=["inputs"])
        return dataset


@rich.repr.auto
class CotV2(TextStrategy):
    def clean(self):
        dataset = self.dropna(columns=["inputs"]).drop_duplicates(columns=["inputs"])
        return dataset


@rich.repr.auto
class CommonSense170k(ColumnStrategy):
    def clean(self):
        dataset = self.dropna(columns=["output"]).drop_duplicates(
            columns=["instruction"]
        )
        return dataset


@rich.repr.auto
class AyaXlelWd(TextStrategy):
    def clean(self):
        dataset = (
            self.dropna(columns=["inputs"])
            .drop_duplicates(columns=["inputs"])
            .filter(
                lambda example: example["language"] == "tha",
                desc="Filter Thai",
                num_proc=self.num_proc,
            )
        )
        return dataset


@rich.repr.auto
class Aya(TextStrategy):
    def clean(self):
        dataset = (
            self.dropna(columns=["inputs"])
            .drop_duplicates(columns=["inputs"])
            .filter(
                lambda example: example["language"] == "tha",
                desc="Filter Thai",
                num_proc=self.num_proc,
            )
        )
        return dataset


@rich.repr.auto
class Alt(ColumnStrategy):
    def clean(self):
        dataset = (
            self.dropna(columns=["translation"])
            .drop_duplicates(columns=["translation"])
            .map(
                self.translate_columns,
                desc="Translate columns",
                fn_kwargs={"lang_cols": ["en", "th"]},
                num_proc=self.num_proc,
            )
        )
        return dataset
