from setuptools import setup, find_packages

setup(
    name="knowledge-hybrid",
    version="1.0.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "click",
        "FlagEmbedding",
        "llama-index-embeddings-openai",
        "llama-index-readers-file",
        "llama-index-vector-stores-milvus==0.2.4",
        "llama-index-embeddings-ollama",
        "llama-index-core",
        "peft",
        "Pillow",
        "pymilvus",
        "python-pptx",
        "rich",
        "torch",
        "transformers",
        "Wand",
    ],
    entry_points={
        "console_scripts": [
            "knowledge-hybrid=knowledge.cli:cli",
        ],
    },
)
