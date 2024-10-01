import click
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from knowledge.embeddings.sparse_embedding import SparseEmbedding
import json


@click.command("retrieve")
@click.option(
    "-d", "--dataset", required=True, help="Name of the dataset to retrieve from"
)
@click.option(
    "--uri",
    default="http://localhost:19530",
    help="Milvus server URI. (default: http://localhost:19530)",
)
@click.option(
    "-n",
    "--num-results",
    default=5,
    type=int,
    help="Number of results to return (default: 5)",
)
@click.option(
    "-k",
    "--top-k",
    default=10,
    type=int,
    help="Number of sources to retrieve (default: 10)",
)
@click.argument("query", required=True)
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
def retrieve_dataset(
    dataset,
    uri,
    num_results,
    top_k,
    query,
    embedding_model,
    openai_embedding_model,
    ollama_embedding_model,
):
    """
    Retrieve information from a dataset based on a query.

    QUERY: The query string to search for in the dataset.
    """
    try:
        vector_store = MilvusVectorStore(
            collection_name=dataset,
            uri=uri,
            enable_sparse=True,
            sparse_embedding_function=SparseEmbedding(),
            hybrid_ranker="RRFRanker",
            hybrid_ranker_params={"k": 60},
        )

        if embedding_model == "ollama":
            embed_model = OllamaEmbedding(
                model_name=ollama_embedding_model,
                base_url="http://localhost:11434",
                ollama_additional_kwargs={"mirostat": 0},
            )
        else:
            embed_model = OpenAIEmbedding(model=openai_embedding_model)

        index = VectorStoreIndex.from_vector_store(vector_store)
        index._embed_model = embed_model

        query_engine = index.as_retriever(similarity_top_k=top_k)
        response = query_engine.retrieve(query)
        to_return = []
        for resp in response:
            to_return.append(
                {
                    "id": resp.id_,
                    "file_path": resp.metadata["file_path"],
                    "content": resp.text,
                    "similarity_score": resp.score,
                }
            )

        click.echo(f"Query: {query}")
        click.echo("Results:")
        for i, result in enumerate(to_return[:num_results], 1):
            # click.echo(
            #     f"--- begin document --- {i}. Source: {result['file_path']} | Score: {result['similarity_score']:.4f} | Content: {result['content']}... --- end document ---"
            # )
            click.echo(json.dumps(result))

    except Exception as e:
        click.echo(f"Failed to retrieve from dataset '{dataset}': {str(e)}", err=True)
