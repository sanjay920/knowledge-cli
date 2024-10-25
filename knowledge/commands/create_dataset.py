import click
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.milvus import MilvusVectorStore
from knowledge.embeddings.sparse_embedding import SparseEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding

# Core logic separated into a callable function
def create_new_dataset(
    collection_name,
    dimension=1536,  # Default dimension of vector embeddings
    uri="http://localhost:19530",  # Default Milvus server URI
    overwrite=False,  # Default is not to overwrite existing collection
    enable_sparse=True,  # Default is to enable sparse vector storage
    hybrid_k=60,  # Default K value for hybrid ranker
    embedding_model="openai",  # Default embedding model
    openai_embedding_model="text-embedding-3-small",  # Default OpenAI embedding model
    ollama_embedding_model="jina/jina-embeddings-v2-base-en",  # Default Ollama embedding model
):
    """
    Create a new dataset using LlamaIndex and Milvus.
    This is the core function that can be called programmatically.
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

        return f"Dataset '{collection_name}' created successfully."
    
    except Exception as e:
        return f"Failed to create dataset '{collection_name}': {str(e)}"


# The Click-decorated version for CLI use
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
    help="OpenAI model name. (default: text-embedding-3-small)",
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
    result = create_new_dataset(
        collection_name,
        dimension,
        uri,
        overwrite,
        enable_sparse,
        hybrid_k,
        embedding_model,
        openai_embedding_model,
        ollama_embedding_model,
    )
    click.echo(result)
