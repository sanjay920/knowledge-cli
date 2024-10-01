import click
from knowledge.commands.create_dataset import create_dataset
from knowledge.commands.delete_dataset import delete_dataset
from knowledge.commands.ingest_dataset import ingest_dataset
from knowledge.commands.retrieve_dataset import retrieve_dataset
from knowledge.commands.get_datasets import get_datasets


@click.group()
@click.version_option(version="1.0.0", prog_name="knowledge")
def cli():
    """
    Knowledge CLI - Manage datasets and retrieve knowledge.
    """
    pass


# Register subcommands
cli.add_command(create_dataset)
cli.add_command(delete_dataset)
cli.add_command(ingest_dataset)
cli.add_command(retrieve_dataset)
cli.add_command(get_datasets)
