import click
from knowledge.datastore.client import get_client


@click.command("get-datasets")
@click.option(
    "--uri",
    default="http://localhost:19530",
    help="Milvus server URI. (default: http://localhost:19530)",
)
def get_datasets(uri):
    """
    Get all datasets (collections) from the Knowledge Base.
    """
    try:
        # Get Milvus client
        client = get_client(uri=uri)

        # List all collections
        collections = client.list_collections()

        if collections:
            click.echo("Available datasets:")
            for collection in collections:
                click.echo(f"- {collection}")
        else:
            click.echo("No datasets found.")
    except Exception as e:
        click.echo(f"Failed to retrieve datasets: {str(e)}", err=True)
