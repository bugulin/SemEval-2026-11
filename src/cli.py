import json
import tempfile
from io import Reader, Writer
from os.path import join

import click
from huggingface_hub import login


@click.group()
@click.option("--hugging-face-token", envvar="HG_TOKEN")
def cli(hugging_face_token: str | None):
    if hugging_face_token is not None:
        login(hugging_face_token)


@cli.command()
@click.option(
    "-n",
    type=int,
    default=500,
    help="Set the number of syllogisms to generate.",
)
def generate(n: int):
    """Generate syllogisms in the specified format."""
    from generator import generate_syllogisms

    generate_syllogisms(n)


@cli.command()
@click.option(
    "-d",
    "--dataset",
    type=click.Path(exists=True),
    multiple=True,
    help="Add training dataset.",
)
@click.option(
    "--preprocess/--no-preprocess",
    default=True,
    help="Transform the dataset to fit the instruct structure.",
)
@click.option(
    "-m",
    "--model",
    type=str,
    default="meta-llama/Meta-Llama-3.1-8B-Instruct",
    help="Set the name of the model to be fine-tuned.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(file_okay=False),
    default="./syllogism_model_checkpoints",
    help="Set the path to output directory.",
)
def fine_tune(dataset: list[str], preprocess: bool, model: str, output: str):
    """Fine-tune an existing model."""
    from training.lora import fine_tune, prepare_dataset

    if preprocess:
        with tempfile.TemporaryDirectory(dir=".tmp-data") as tmp:
            datasets = [
                prepare_dataset(ds, join(tmp, f"{i:02d}.json"))
                for i, ds in enumerate(dataset)
            ]
            fine_tune(model, datasets, output)
    else:
        fine_tune(model, dataset, output)


@cli.command()
@click.option(
    "-m",
    "--model",
    type=str,
    default="meta-llama/Llama-3.1-8B-Instruct",
    help="Change the language model used.",
)
@click.option(
    "-a",
    "--adapter",
    type=str,
    default="MatusZelko/llama-3.1-syllogism-lora",
    help="Change the adapter used.",
)
@click.argument("file", type=click.File("r"))
def run(model: str, adapter: str, file: Reader[str]):
    from inference import PeftSyllogismSolver

    lm = PeftSyllogismSolver(model, adapter)
    responses = [response for response in lm.solve(json.load(file))]
    click.echo_via_pager(json.dumps(responses))


@cli.command()
@click.option(
    "-r",
    "--ref",
    type=click.Path(dir_okay=False, readable=True),
    help="Set the path to reference data.",
)
@click.option(
    "-o",
    "--out",
    type=click.File("w"),
    help="Write output to the file.",
    default="-",
)
@click.argument("prediction", nargs=-1)
def evaluate(ref: str, prediction: tuple[str, ...], out: Writer[str]):
    from evaluation import run_full_scoring

    for pred in prediction:
        run_full_scoring(ref, pred, out)


if __name__ == "__main__":
    cli()
