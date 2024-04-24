from argparse import ArgumentTypeError
from argparse import ArgumentParser
from rich_argparse import RichHelpFormatter
from rich.markdown import Markdown

LOGO = """
⠀_⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀_⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀_⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀_⠀⠀⠀⠀_⠀⠀
 \ \        / /                   | |                 \ \ / /⠀⠀
  \ \  /\  / /_ _ _ __   __ _  ___| |__   __ _ _ __    \ V / ⠀⠀
   \ \/  \/ / _` | '_ \ / _` |/ __| '_ \ / _` | '_ \    > <  ⠀⠀
    \  /\  / (_| | ( | | (_| | (__| ( | | (_| | ( | |  / . \ ⠀⠀
     \/  \/ \__,_|_| |_|\__, |\___|_| |_|\__,_|_| |_| /_/ \_\ ⠀⠀
                         __/ |                               ⠀⠀
                        |___/                                ⠀⠀

         [bold magenta]║[/bold magenta] [bold cyan]▀▄▀▄▀▄[/bold cyan] [bold green] 🛠️ [/bold green] [bold yellow]𝕋 𝕆 𝕆 𝕃 𝕂 𝕀 𝕋 𝕊[/bold yellow] [bold green] 🛠️ [/bold green] [bold cyan]▄▀▄▀▄▀[/bold cyan] [bold magenta]║[/bold magenta]
"""


DESC = "This script creates a FLAN (Finetuned Language Models Are Zero-Shot Learners) like dataset by combining various datasets from different sources. The script loads datasets from HuggingFace, custom datasets, and performs data processing and reformatting to prepare the data for training."


class ArgparseType:
    @staticmethod
    def positive_number(value):
        try:
            number = int(value)
            if number <= 0:
                raise ArgumentTypeError(f"Expected positive integer, got {value!r}")
            return number
        except ValueError:
            raise ArgumentTypeError(f"Expected integer, got {value!r}")

    @staticmethod
    def boolean(value):
        try:
            return eval(value.capitalize())
        except NameError:
            raise ArgumentTypeError(f"Expected boolean, got {value!r}")


def get_parser(description=DESC) -> ArgumentParser:
    _parser = ArgumentParser(
        description=Markdown(description, style="argparse.text"),
        formatter_class=RichHelpFormatter,
    )
    _parser.add_argument(
        "--metadata",
        "-m",
        type=str,
        default="default",
        help="Path to the JSON file containing the metadata for creating the FLAN dataset.",
    )
    _parser.add_argument(
        "--template",
        "-t",
        type=str,
        default="default",
        help="Path to the JSON file containing the template for creating the FLAN dataset.",
    )
    _parser.add_argument(
        "--cache_dir",
        type=str,
        default="./cache",
        help="Directory to cache the downloaded datasets",
    )
    _parser.add_argument(
        "--num_proc",
        default=1,
        type=ArgparseType.positive_number,
        help="Number of processes to use for parallel processing. Set to a value greater than 1 to enable multiprocessing. Default is 1 (no parallelization). Adjust this value based on the available CPU cores and memory.",
    )
    _parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        required=True,
        help="Directory to save the resulting FLAN dataset",
    )
    _parser.add_argument(
        "--metafile",
        "-mf",
        type=str,
        default="./metadata/metafile.txt",
        help="Path to save the dataset metadata",
    ),
    _parser.add_argument(
        "-c",
        "--customize_dataset_size",
        nargs=2,
        metavar=("DATASET_NAME", "SAMPLE_COUNT"),
        help="Customize the number of samples for a specific dataset. Can be used multiple times. Provide the dataset name and the desired sample count (positive integer).",
        action="append",
        type=lambda x: (
            (str(x[0]), ArgparseType.positive_number(x[1]))
            if isinstance(x, list)
            else x
        ),
    ),
    _parser.add_argument(
        "-s",
        "--save_format",
        nargs=1,
        help="Save the dataset in the specified format. Can be 'json', 'jsonl', 'csv', or 'arrow'.",
        choices=["jsonl", "csv", "arrow"],
        default="arrow",
    ),
    _parser.add_argument(
        "-d",
        "--drop_mode",
        nargs=1,
        help="Drop duplicates mode. Can be 'huggingface' or 'pandas'.",
        choices=["huggingface", "pandas", "none"],
        default="none",
    ),

    return _parser
