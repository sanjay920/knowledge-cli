import click
from knowledge.datastore.client import get_client


# Core logic for deleting a dataset
def delete_datasets(collection_name, uri="http://localhost:19530", force=False):
    """
    Delete a dataset from Milvus programmatically.

    Args:
        collection_name (str): The name of the dataset (collection) to delete.
        uri (str): Milvus server URI (default: http://localhost:19530).
        force (bool): Whether to force deletion without confirmation (default: False).
    """
    try:
        # Get Milvus client
        client = get_client(uri=uri)

        # Check if the collection exists
        if not client.has_collection(collection_name):
            return f"Dataset '{collection_name}' does not exist."

        # If force is not enabled, ask for confirmation
        if not force:
            confirm = input(f"Are you sure you want to delete the dataset '{collection_name}'? (y/n): ")
            if confirm.lower() != 'y':
                return "Deletion cancelled."

        # Delete the collection
        client.drop_collection(collection_name)

        return f"Dataset '{collection_name}' deleted successfully."
    except Exception as e:
        return f"Failed to delete dataset '{collection_name}': {str(e)}"


# CLI command using Click
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
    result = delete_datasets(collection_name, uri, force)
    click.echo(result)
