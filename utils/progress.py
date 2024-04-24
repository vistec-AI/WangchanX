from .argparse_helper import LOGO
from rich.console import Group
from rich.panel import Panel
from rich.live import Live
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.layout import Layout
from rich.align import Align
import pandas as pd
from rich.table import Table
from rich.json import JSON


current_app_progress = Progress(
    TimeElapsedColumn(),
    TextColumn("{task.description}"),
)

step_progress = Progress(
    TextColumn("  "),
    TimeElapsedColumn(),
    TextColumn("[bold purple]{task.fields[action]}"),
    SpinnerColumn("simpleDots"),
)


app_steps_progress = Progress(
    TextColumn("[bold blue]Progress for {task.fields[name]}: {task.percentage:.0f}%"),
    BarColumn(),
    TextColumn("({task.completed} of {task.total} steps done)"),
)

overall_progress = Progress(
    TimeElapsedColumn(), BarColumn(bar_width=None), TextColumn("{task.description}")
)

progress_group = Group(
    Panel(
        Group(
            current_app_progress,
            step_progress,
            app_steps_progress,
        ),
        title="Steps",
    ),
    overall_progress,
)


main_layout = Layout()
main_layout.split_column(
    Layout(
        Align.center(LOGO),
        name="logo",
    ),
    Layout(
        Panel("", title="Examples"),
        name="examples",
    ),
    Layout(progress_group, name="progress"),
)
main_layout["logo"].size = 13
main_layout["examples"].size = 24


def df_to_table(
    pandas_dataframe: pd.DataFrame,
    rich_table: Table,
    show_index: bool = True,
    index_name: str | None = None,
) -> Table:

    if show_index:
        index_name = str(index_name) if index_name else ""
        rich_table.add_column(index_name)

    for column in pandas_dataframe.columns:
        rich_table.add_column(str(column))

    for index, value_list in enumerate(pandas_dataframe.values.tolist()):
        row = [str(index)] if show_index else []
        row += [str(x) for x in value_list]
        rich_table.add_row(*row)

    return rich_table


def run_steps(name, step_times, app_steps_task_id, dataset_manager):
    """Run steps for a single app, and update corresponding progress bars."""

    for action in step_times:
        step_task_id = step_progress.add_task("", action=action, name=name)

        if action == "downloading":
            dataset_manager._download(name)
            step_progress.update(step_task_id, advance=1)

        elif action == "cleaning":
            dataset_manager.clean(name)
            step_progress.update(step_task_id, advance=1)

        elif action == "formatting":
            dataset_manager.reformat(name)
            main_layout["examples"].update(
                Panel(
                    JSON.from_data(dataset_manager.get_examples(name)),
                    title=f"Examples {name}",
                )
            )
            step_progress.update(step_task_id, advance=1)

        elif action == "drop duplicates":
            dataset_manager.drop_duplicates()
            step_progress.update(step_task_id, advance=1)

        elif action == "save":
            dataset_manager.save_datasets()

        step_progress.stop_task(step_task_id)
        step_progress.update(step_task_id, visible=False)

        app_steps_progress.update(app_steps_task_id, advance=1)


def run_apps(apps, console, dataset_manager):
    overall_task_id = overall_progress.add_task("", total=len(apps))
    with Live(main_layout, console=console):
        for idx, (name, step_times) in enumerate(apps):
            top_descr = "[bold #AAAAAA](%d out of %d processed)" % (
                idx,
                len(apps),
            )
            overall_progress.update(overall_task_id, description=top_descr)

            current_task_id = current_app_progress.add_task("Processing %s" % name)
            app_steps_task_id = app_steps_progress.add_task(
                "", total=len(step_times), name=name
            )
            run_steps(name, step_times, app_steps_task_id, dataset_manager)

            app_steps_progress.update(app_steps_task_id, visible=False)
            current_app_progress.stop_task(current_task_id)
            current_app_progress.update(
                current_task_id, description="[bold green]%s done!" % name
            )

            overall_progress.update(overall_task_id, advance=1)

        overall_progress.update(
            overall_task_id,
            description="[bold green]%s processed, done!" % len(apps),
        )
    console.clear()
    table = dataset_manager.save_metadata()
    table = Align.center(df_to_table(table, Table(title="Metadata"), show_index=False))
    console.print(Align.center(LOGO))
    console.print("\n")
    console.print(table)
    console.print("\n")
    console.print(Panel(current_app_progress, title="Steps"))
    console.print(overall_progress)
