import click
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.milvus import MilvusVectorStore
from knowledge.embeddings.sparse_embedding import SparseEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding


@click.command("create-dataset")
@click.argument("collection_name", type=str, required=True)
@click.option(
    "--dimension",
    "-d",
    default=1536,
    help="Dimension of the vector embeddings. (default: 1536)",
)
@click.option(
    "--uri",
    default="http://localhost:19530",
    help="Milvus server URI. (default: http://localhost:19530)",
)
@click.option(
    "--overwrite",
    default=False,
    is_flag=True,
    help="Overwrite existing collection if it exists.",
)
@click.option(
    "--enable-sparse",
    is_flag=True,
    default=True,
    help="Enable sparse vector storage. (default: True)",
)
@click.option("--hybrid-k", default=60, help="K value for hybrid ranker. (default: 60)")
@click.option(
    "--embedding-model",
    type=click.Choice(["openai", "ollama"]),
    default="openai",
    help="Embedding model to use. (default: openai)",
)
@click.option(
    "--openai-embedding-model",
    default="text-embedding-3-small",
    help="Ollama model name. (default: text-embedding-3-small)",
)
@click.option(
    "--ollama-embedding-model",
    default="jina/jina-embeddings-v2-base-en",
    help="Ollama model name. (default: jina/jina-embeddings-v2-base-en)",
)
def create_dataset(
    collection_name,
    dimension,
    uri,
    overwrite,
    enable_sparse,
    hybrid_k,
    embedding_model,
    openai_embedding_model,
    ollama_embedding_model,
):
    """
    Create a new dataset using LlamaIndex and Milvus.

    COLLECTION_NAME: The name of the collection to create.
    """
    try:
        vector_store = MilvusVectorStore(
            dim=dimension,
            collection_name=collection_name,
            overwrite=overwrite,
            uri=uri,
            enable_sparse=enable_sparse,
            sparse_embedding_function=SparseEmbedding(),
            hybrid_ranker="RRFRanker",
            hybrid_ranker_params={"k": hybrid_k},
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

        VectorStoreIndex.from_vector_store(
            vector_store, storage_context=storage_context, embed_model=embed_model
        )

        click.echo(f"Dataset '{collection_name}' created successfully.")
    except Exception as e:
        click.echo(f"Failed to create dataset '{collection_name}': {str(e)}", err=True)
