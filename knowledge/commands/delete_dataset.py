import click
from knowledge.datastore.client import get_client


@click.command("delete-dataset")
@click.argument("collection_name", type=str, required=True)
@click.option(
    "--uri",
    default="http://localhost:19530",
    help="Milvus server URI. (default: http://localhost:19530)",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Force deletion without confirmation.",
)
def delete_dataset(collection_name, uri, force):
    """
    Delete a dataset from Milvus.

    COLLECTION_NAME: The name of the collection to delete.
    """
    try:
        # Get Milvus client
        client = get_client(uri=uri)

        # Check if the collection exists
        if not client.has_collection(collection_name):
            click.echo(f"Dataset '{collection_name}' does not exist.", err=True)
            return

        if not force:
            confirm = click.confirm(
                f"Are you sure you want to delete the dataset '{collection_name}'?"
            )
            if not confirm:
                click.echo("Deletion cancelled.")
                return

        # Delete the collection
        client.drop_collection(collection_name)

        click.echo(f"Dataset '{collection_name}' deleted successfully.")
    except Exception as e:
        click.echo(f"Failed to delete dataset '{collection_name}': {str(e)}", err=True)
