from utils.argparse_helper import get_parser, LOGO
from utils.progress import run_apps
from rich.console import Console
from flan_creator import DatasetManager
import json
import os
from rich.pretty import pprint
from datasets.utils.logging import disable_progress_bar
import sys


if "HF_DATASETS_DISABLE_PROGRESS_BAR" not in os.environ:
    disable_progress_bar()

CURRENT_DIR = os.path.dirname(__file__)

with open(f"{CURRENT_DIR}/flan_creator/dataset_info.json", "r") as file:
    DEFAULT_DATASET_INFO = json.load(file)

with open(f"{CURRENT_DIR}/data_formatter/templates.json", "r") as file:
    DEFAULT_TEMPLATE = json.load(file)


CONSOLE = Console()


def check_required_keys(data):
    required_keys = {
        "template",
        "source",
        "task",
        "license",
        "domain",
        "processor",
        "num_examples",
        "file_format",
    }

    missing_keys = {}

    for name, entry in data.items():
        entry_missing_keys = required_keys - set(entry.keys())
        if entry_missing_keys:
            missing_keys[name] = list(entry_missing_keys)
        if "source" in entry and not isinstance(entry["source"], dict):
            missing_keys.setdefault(name, []).append("source.path")
        elif "source" in entry and "path" not in entry["source"]:
            missing_keys.setdefault(name, []).append("source.path")

    return missing_keys


def convert_k(k_str):
    k_str = k_str.lower().strip()

    if k_str in ["all", "half", "quarter"]:
        return k_str

    try:
        k = int(k_str)
        if k > 0:
            return k
        else:
            CONSOLE.print("k must be a positive integer.", style="bold red")
            sys.exit(1)
    except ValueError:
        CONSOLE.print("k must be a positive integer.", style="bold red")
        sys.exit(1)


def load_json_file(file_path):
    if not os.path.exists(file_path):
        CONSOLE.print(f"File {file_path} not found!!", style="bold red")
        sys.exit(1)
    with open(file_path, "r") as file:
        return json.load(file)


def ensure_directory_exists(path):
    directory_path = os.path.dirname(path)
    if directory_path and not os.path.exists(directory_path):
        os.makedirs(directory_path)


def update_dataset_info(dataset_info, customizations):
    for dataset_name, sample_count in customizations:
        if dataset_name not in dataset_info:
            CONSOLE.print(f"Dataset {dataset_name} not found!!", style="bold red")
            sys.exit(1)
        dataset_info[dataset_name]["num_examples"] = convert_k(sample_count)


def check_txt_path(path):

    if not path.lower().endswith(".txt"):
        CONSOLE.print(
            "Saving metadata path must end with '.txt'. Please provide a .txt file path.",
            style="bold red",
        )
        sys.exit(1)
    return True


def get_drop_mode(args_drop_mode):
    return None if args_drop_mode == "none" else args_drop_mode


def create_apps_list(dataset_info, drop_mode):
    apps = [(name, ("downloading", "cleaning", "formatting")) for name in dataset_info]
    if drop_mode in ["huggingface", "pandas"]:
        apps.append(("Drop Duplicates", ("drop duplicates",)))
    apps.append(("Save Dataset", ("save",)))
    return apps


def main():
    parser = get_parser()
    args = parser.parse_args()

    dataset_info = DEFAULT_DATASET_INFO
    if args.metadata != "default":
        dataset_info = load_json_file(args.metadata) if args.metadata else {}
    if check_required_keys(dataset_info):
        CONSOLE.print(
            f"Missing keys in metadata file: {check_required_keys(dataset_info)}",
            style="bold red",
        )
        sys.exit(1)

    template = DEFAULT_TEMPLATE
    if args.template != "default":
        template = load_json_file(args.template) if args.template else {}

    ensure_directory_exists(args.cache_dir)

    if args.metafile:
        if not check_txt_path(args.metafile):
            CONSOLE.print("[bold red]Error:[/bold red] Invalid metafile path. Exiting.")
            return
        ensure_directory_exists(os.path.dirname(args.metafile))

    if args.customize_dataset_size:
        update_dataset_info(dataset_info, args.customize_dataset_size)

    drop_mode = get_drop_mode(args.drop_mode[0].lower())

    dataset_manager = DatasetManager(
        templates=template,
        dataset_metadata=dataset_info,
        cache_dir=args.cache_dir,
        save_dir=args.output_dir,
        save_format=args.save_format,
        save_metadata=args.metafile,
        num_proc=args.num_proc,
        drop_mode=drop_mode,
    )

    apps = create_apps_list(dataset_info, drop_mode)
    run_apps(apps, CONSOLE, dataset_manager)


if __name__ == "__main__":
    main()
