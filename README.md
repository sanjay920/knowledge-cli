# Knowledge CLI

Knowledge CLI is a command-line tool for managing and querying datasets using vector embeddings and semantic search capabilities.

## Features

- Create, delete, and manage datasets
- Ingest data from files or directories
- Retrieve information from datasets using semantic queries
- Support for multiple embedding models (OpenAI and Ollama)
- Hybrid search capabilities (dense and sparse embeddings)

## Prerequisites

### OpenAI API Key

Set your OpenAI API Key

```bash
export OPENAI_API_KEY=YOUR_API_KEY
```

### Milvus Setup

This CLI tool requires Milvus to be running. The easiest way to set up Milvus is using Docker. Follow these steps to install and run Milvus:

1. Make sure you have Docker installed on your system.

2. Run the following commands to download and start Milvus:

   ```bash
   curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh
   bash standalone_embed.sh start
   ```

   This will download and start a standalone Milvus instance using Docker.

3. Wait for Milvus to start up. You should see a message indicating that Milvus is ready.

4. You can now use the Knowledge CLI tool, which will connect to this Milvus instance by default.

To stop Milvus when you're done, you can use:

```bash
bash standalone_embed.sh stop
```

For more information on Milvus setup and configuration, please refer to the [Milvus documentation](https://milvus.io/docs/install_standalone-docker.md).

## Installation

To install the Knowledge CLI, follow these steps:

1. Clone the repository:

   ```
   git clone https://github.com/sanjay920/knowledge-cli.git
   cd knowledge-cli
   ```

2. Install the package in editable mode:

   ```
   pip install -e .
   ```

This will install the `knowledge-hybrid` command-line tool and its dependencies.

## Usage

After installation, you can use the `knowledge-hybrid` command to interact with the CLI. Here are some example commands:

### Create a dataset

```bash
knowledge-hybrid create-dataset my_dataset
```

### Ingest data into a dataset

```bash
knowledge-hybrid ingest -d my_dataset /path/to/data
```

### Retrieve information from a dataset

```bash
knowledge-hybrid retrieve -d my_dataset -k 10 "Your query here"
```

### List all datasets

```bash
knowledge-hybrid get-datasets
```

### Delete a dataset

```bash
knowledge-hybrid delete-dataset my_dataset
```

## Usage with GPTScript

Make sure you ingest a dataset first, then you can run the following to start chatting:

```bash
gptscript quickstart.gpt
```

## Configuration

The CLI uses Milvus as its vector database. By default, it connects to a local Milvus instance at `http://localhost:19530`. You can modify this by using the `--uri` option in relevant commands.

**Note:** Make sure Milvus is running before using the Knowledge CLI tool. See the "Prerequisites" section for instructions on setting up Milvus.

## Embedding Models

The CLI supports two embedding models:

1. OpenAI (default)
2. Ollama

You can specify the embedding model using the `--embedding-model` option in relevant commands.

## Development

To set up the development environment:

1. Create a virtual environment:

   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

2. Install development dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Install the package in editable mode:

   ```
   pip install -e .
   ```
