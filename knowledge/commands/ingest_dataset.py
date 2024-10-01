import click
import os
from knowledge.datastore.client import get_client
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    Document,
)
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.node_parser import TokenTextSplitter
from collections import defaultdict
from tqdm import tqdm
from knowledge.embeddings.sparse_embedding import SparseEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding


def ingest_directory(
    source: str,
    collection_name: str,
    uri: str,
    embedding_model: str,
    openai_embedding_model: str,
    ollama_embedding_model: str,
) -> None:
    reader = SimpleDirectoryReader(
        source,
        recursive=True,
        required_exts=[".pptx", ".pdf"],
    )
    grouped_docs = defaultdict(list)
    for docs in tqdm(
        reader.iter_data(),
        desc="Processing documents",
        total=len(reader.list_resources()),
    ):
        for doc in docs:
            file_path = doc.metadata.get("file_path")
            grouped_docs[file_path].append(doc)

    combined_docs = []
    for file_path, docs in grouped_docs.items():
        # Sort documents by page number
        sorted_docs = sorted(
            docs,
            key=lambda x: (
                int(x.metadata.get("page_label", "0"))
                if x.metadata.get("page_label", "0").isdigit()
                else 0
            ),
        )

        # Combine text from all pages
        combined_text = "\n\n".join(doc.text for doc in sorted_docs if doc.text.strip())

        # Create a new document with combined text
        combined_doc = Document(
            text=combined_text,
            metadata={
                "file_path": file_path,
                "file_name": sorted_docs[0].metadata.get("file_name"),
                "file_type": sorted_docs[0].metadata.get("file_type"),
                "file_size": sorted_docs[0].metadata.get("file_size"),
                "creation_date": sorted_docs[0].metadata.get("creation_date"),
                "last_modified_date": sorted_docs[0].metadata.get("last_modified_date"),
                "total_pages": len(sorted_docs),
            },
        )
        combined_docs.append(combined_doc)

    try:
        vector_store = MilvusVectorStore(
            dim=1536,  # TODO: Hardcoding for now
            collection_name=collection_name,
            overwrite=False,
            uri=uri,
            enable_sparse=True,  # TODO: Hardcoding for now
            sparse_embedding_function=SparseEmbedding(),
            hybrid_ranker="RRFRanker",
            hybrid_ranker_params={"k": 60},  # TODO: Hardcoding for now
        )
        if embedding_model == "ollama":
            embed_model = OllamaEmbedding(
                model_name=ollama_embedding_model,
                base_url="http://localhost:11434",
                ollama_additional_kwargs={"mirostat": 0},
            )
        else:
            embed_model = OpenAIEmbedding(model=openai_embedding_model)

        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            combined_docs,
            storage_context=storage_context,
            show_progress=True,
            transformations=[
                TokenTextSplitter(separator=" ", chunk_size=1024, chunk_overlap=256)
            ],
            embed_model=embed_model,
        )
    except Exception as e:
        click.echo(f"Failed to index data: {str(e)}", err=True)


@click.command("ingest")
@click.option(
    "-d", "--dataset", required=True, help="Name of the dataset to ingest into"
)
@click.argument("source", type=click.Path(exists=True), required=True)
@click.option(
    "--uri",
    default="http://localhost:19530",
    help="Milvus server URI. (default: http://localhost:19530)",
)
@click.option(
    "--embedding-model",
    type=click.Choice(["openai", "ollama"]),
    default="openai",
    help="Embedding model to use. (default: openai)",
)
@click.option(
    "--openai-embedding-model",
    default="text-embedding-3-small",
    help="OpenAI embedding model name. (default: text-embedding-3-small)",
)
@click.option(
    "--ollama-embedding-model",
    default="jina/jina-embeddings-v2-base-en",
    help="Ollama embedding model name. (default: jina/jina-embeddings-v2-base-en)",
)
def ingest_dataset(
    dataset,
    source,
    uri,
    embedding_model,
    openai_embedding_model,
    ollama_embedding_model,
):
    """
    Ingest data from a directory into an existing dataset.

    SOURCE: The directory to ingest.
    """
    try:
        # Get Milvus client
        client = get_client(uri=uri)

        # Check if the dataset (collection) exists
        if not client.has_collection(dataset):
            click.echo(f"Dataset '{dataset}' does not exist.", err=True)
            return

        print(source)

        # Check if the source is a file or directory
        if os.path.isfile(source):
            click.echo(
                f"Invalid source: {source}. Cannot provide a file as a source.",
                err=True,
            )
            return
        elif os.path.isdir(source):
            click.echo(f"Ingesting directory: {source}")
            # TODO: Add logic for ingesting a directory
            ingest_directory(
                source,
                dataset,
                uri,
                embedding_model,
                openai_embedding_model,
                ollama_embedding_model,
            )
        else:
            click.echo(
                f"Invalid source: {source}. The specified file or directory does not exist.",
                err=True,
            )
            return

        click.echo(f"Data ingestion into dataset '{dataset}' completed successfully.")
    except Exception as e:
        click.echo(
            f"Failed to ingest data into dataset '{dataset}': {str(e)}", err=True
        )
