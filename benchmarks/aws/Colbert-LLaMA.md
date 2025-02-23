This repository provides an implementation of a **Retrieval-Augmented Generation (RAG)** system tailored for biomedical question answering. The system uses a **ColBERT** model for document embeddings and the **LLaMA2 34B model** for generating answers. The goal is to combine the power of retrieval-based approaches with generative models to answer complex biomedical queries from the **BioASQ** dataset. The system runs on **Amazon SageMaker** with different storage configurations, such as **NVMe on EC2**, **EBS**, and **DRAM**, for efficient indexing and retrieval.

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Dataset Preparation](#dataset-preparation)
   - [BioASQ Dataset](#bioasq-dataset)
4. [Environment Setup](#environment-setup)
5. [Indexing and Retrieval](#indexing-and-retrieval)
   - [ColBERT Embedding Model](#colbert-embedding-model)
   - [Creating the HNSW Index](#creating-the-hnsw-index)
   - [Storage Configurations](#storage-configurations)
6. [Model Deployment](#model-deployment)
   - [Deploying LLaMA2 34B Model](#deploying-llama2-34b-model)
7. [Pipeline Execution](#pipeline-execution)
   - [Processing BioASQ Queries](#processing-bioasq-queries)
   - [Profiling and Performance Metrics](#profiling-and-performance-metrics)
8. [Performance Statistics](#performance-statistics)
9. [Conclusion](#conclusion)

## Overview

This implementation sets up a **Retrieval-Augmented Generation (RAG)** pipeline for biomedical question answering (QA). It utilizes **ColBERT** to generate dense document embeddings and a **LLaMA2 34B model** to generate answers based on retrieved documents. The process involves embedding biomedical text, creating an **HNSW** (Hierarchical Navigable Small World) index for fast retrieval, and deploying LLaMA2 for answer generation.

Performance profiling, including latency and throughput metrics for each stage of the system (query embedding, retrieval, augmentation, and generation), is provided. This allows users to evaluate the system’s efficiency and scalability.

## Prerequisites

- AWS account with permissions for SageMaker, EC2, and S3 services.
- Python 3.x installed.
- AWS CLI and Boto3 configured.
- Basic knowledge of machine learning and cloud services.

### Install Required Libraries

```bash
pip install faiss-cpu transformers datasets boto3 colbert
```

## Dataset Preparation

### BioASQ Dataset

The **BioASQ** dataset provides benchmark datasets for biomedical question answering. You will need the QA-task10bPhaseA test set for this implementation.

Download it with the following:

```bash
wget http://participants-area.bioasq.org/Tasks/10b/QA-task10bPhaseA-testset4.json
```

The BioASQ dataset contains **questions** and **answers**, which will be used to query the system.

## Environment Setup

1. **Create an AWS SageMaker Notebook Instance**:
   - Navigate to SageMaker in the AWS console and create a new notebook instance.

2. **Install Required Libraries**:
   - After launching the instance, install the necessary libraries via the command:
   ```bash
   pip install faiss-cpu transformers datasets boto3 colbert
   ```

## Indexing and Retrieval

### ColBERT Embedding Model

We use the **ColBERT** model for embedding biomedical text documents. ColBERT is a fast and efficient bi-encoder model designed for dense retrieval tasks.

To load and use the ColBERT model:

```python
from colbert import ColBERT
from transformers import AutoTokenizer

# Load tokenizer and ColBERT model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = ColBERT.from_pretrained("bert-base-uncased")

# Function to embed documents
def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    embeddings = model(**inputs).pooler_output
    return embeddings.detach().numpy()
```

### Creating the HNSW Index

Use **FAISS** to create the HNSW index for document retrieval. This index will allow us to perform fast nearest-neighbor search.

```python
import faiss
import numpy as np

# Initialize the HNSW index
dim = 768  # Dimension of the embeddings
index = faiss.IndexHNSWFlat(dim, 32)
index.hnsw.efConstruction = 200

# Add embeddings to the index
index.add(embedding_vectors)  # embedding_vectors is a NumPy array of document embeddings
faiss.write_index(index, "hnsw_index.faiss")
```

### Storage Configurations

To store the HNSW index, we explore three different setups:

#### NVMe on EC2

```bash
sudo mkfs -t ext4 /dev/nvme0n1
sudo mount /dev/nvme0n1 /mnt/nvme
mv hnsw_index.faiss /mnt/nvme/
```

#### EBS

```bash
sudo mkfs -t ext4 /dev/xvdf
sudo mount /dev/xvdf /mnt/ebs
mv hnsw_index.faiss /mnt/ebs/
```

#### DRAM

For in-memory storage, load the index directly into memory:

```python
index = faiss.read_index("hnsw_index.faiss")
```

## Model Deployment

### Deploying LLaMA2 34B Model

The **LLaMA2 34B** model is deployed on SageMaker to generate answers. This large model requires a powerful instance for inference.

```python
from sagemaker.huggingface import HuggingFaceModel

# Initialize the LLaMA2 34B model
model = HuggingFaceModel(
    model_data="s3://my-bucket/llama2-34b.tar.gz",  # Path to the model
    role="SageMakerRole",
    transformers_version="4.17",
    pytorch_version="1.10",
    py_version="py38"
)

# Deploy the model on a powerful instance
predictor = model.deploy(instance_type="ml.p4d.24xlarge", initial_instance_count=1)
```

## Pipeline Execution

### Processing BioASQ Queries

For each BioASQ query, the following steps occur:

1. **Embed the Query** using ColBERT.
2. **Retrieve the Top-k Documents** from the HNSW index.
3. **Generate the Answer** using the LLaMA2 34B model based on the retrieved documents.

```python
query = "What are the effects of aspirin on heart disease?"
query_embedding = embed_text(query)

# Perform retrieval
D, I = index.search(query_embedding, k=5)  # Retrieve top 5 documents
retrieved_docs = [documents[i] for i in I[0]]

# Generate the answer using LLaMA2
response = predictor.predict({"input": query, "context": retrieved_docs})
```

### Profiling and Performance Metrics

Measure **latency**, **memory utilization**, and **throughput** for each phase:

```python
import time

# Record the start time for query embedding
start = time.time()
query_embedding = embed_text(query)
embedding_time = time.time() - start

# Record retrieval time
start = time.time()
D, I = index.search(query_embedding, k=5)
retrieval_time = time.time() - start

# Record augmentation and generation time
start = time.time()
response = predictor.predict({"input": query, "context": retrieved_docs})
generation_time = time.time() - start

# Print out profiling metrics
print(f"Query Embedding Time: {embedding_time} seconds")
print(f"Retrieval Time: {retrieval_time} seconds")
print(f"Generation Time: {generation_time} seconds")
```

## Performance Statistics

After processing all BioASQ queries, the following performance statistics will be computed for each query:

1. **Query Embedding Latency**
2. **Document Retrieval Latency**
3. **Answer Generation Latency**
4. **Total Latency** (query to answer)
5. **Memory Utilization** for indexing, embedding, and generation.
6. **Throughput** (queries per second).

For example:

```plaintext
Query: What are the effects of aspirin on heart disease?
Query Embedding Latency: 0.45s
Document Retrieval Latency: 1.2s
Generation Latency: 3.5s
Total Latency: 5.15s
Memory Utilization: 12GB
Throughput: 1 query/sec
```
