import click
from knowledge.datastore.client import get_client
# Core function to retrieve datasets
def get_all_datasets(uri="http://localhost:19530"):
    """
    Retrieve all datasets (collections) from the Knowledge Base.
    
    Args:
        uri (str): The Milvus server URI.

    Returns:
        list: A list of available dataset names.
    """
    try:
        # Get Milvus client
        client = get_client(uri=uri)

        # List all collections
        collections = client.list_collections()

        if collections:
            return collections
        else:
            return "No datasets found."
    except Exception as e:
        return f"Failed to retrieve datasets: {str(e)}"


# Click-decorated function for CLI usage
@click.command("get-datasets")
@click.option(
    "--uri",
    default="http://localhost:19530",
    help="Milvus server URI. (default: http://localhost:19530)",
)
def get_datasets(uri):
    collections = get_all_datasets(uri)
    
    if isinstance(collections, list):
        click.echo("Available datasets:")
        for collection in collections:
            click.echo(f"- {collection}")
    else:
        click.echo(collections)
