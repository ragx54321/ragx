## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Dataset Preparation](#dataset-preparation)
   - [BioASQ Dataset](#bioasq-dataset)
4. [Environment Setup](#environment-setup)
5. [Indexing and Retrieval](#indexing-and-retrieval)
   - [GTR Index](#gtr-index)
   - [Storage Configurations](#storage-configurations)
6. [Model Deployment](#model-deployment)
   - [Deploying LLaMA2 34B Model](#deploying-llama2-34b-model)
7. [Pipeline Execution](#pipeline-execution)
   - [Processing BioASQ Queries](#processing-bioasq-queries)
   - [Profiling and Performance Metrics](#profiling-and-performance-metrics)
8. [Performance Statistics](#performance-statistics)
9. [Conclusion](#conclusion)

## Overview

This implementation sets up a **Retrieval-Augmented Generation (RAG)** pipeline for biomedical question answering using the **GTR** model for document retrieval and **LLaMA2 34B** for generative answer creation. The system follows these steps:


**GTR (Global Text Representation)** is a dense vector-based document retrieval model, which represents text as a single vector for efficient similarity search. Unlike traditional sparse methods (e.g., BM25, TF-IDF), GTR uses pre-trained neural networks to generate embeddings that capture the semantics of the document globally. These embeddings can be indexed and retrieved efficiently using libraries like **Faiss** or **Annoy**. GTR performs well in scenarios where the documents contain more complex information and relationships, making it ideal for biomedical question answering tasks.

## Prerequisites

- AWS account with permissions for SageMaker, EC2, and S3 services.
- Python 3.x installed.
- AWS CLI and Boto3 configured.
- Basic knowledge of machine learning, GTR, and cloud services.

### Install Required Libraries

```bash
pip install transformers datasets faiss-cpu gtr boto3
```

## Dataset Preparation

### BioASQ Dataset

The **BioASQ** dataset is a collection of biomedical questions and answers. We use the **QA-task10bPhaseA** test set for this implementation.

Download it from:

```bash
wget http://participants-area.bioasq.org/Tasks/10b/QA-task10bPhaseA-testset4.json
```

This dataset contains the questions and their corresponding answers, which will be used to evaluate the RAG pipeline.

## Environment Setup

1. **Create an AWS SageMaker Notebook Instance**:
   - Navigate to SageMaker in the AWS console and create a new notebook instance.

2. **Install Required Libraries**:
   - After launching the instance, install the necessary libraries via the command:
   ```bash
   pip install transformers datasets faiss-cpu gtr boto3
   ```

## Indexing and Retrieval

### GTR Index

**GTR** embeds documents into dense vectors using pre-trained neural networks. These vectors are then indexed for efficient similarity search. Below is how you can use GTR for document retrieval:

1. **Preprocessing the Documents**: Tokenize the documents and create embeddings using GTR.
2. **Index the Documents**: Use **Faiss** to index the dense document embeddings for efficient retrieval.

```python
from transformers import AutoTokenizer, AutoModel
import faiss
import torch

# Initialize GTR model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
model = AutoModel.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')

# Tokenize and encode the documents
def encode_documents(documents):
    inputs = tokenizer(documents, padding=True, truncation=True, return_tensors="pt")
    embeddings = model(**inputs).pooler_output
    return embeddings

# Create embeddings for the documents
documents = ["Document 1 text", "Document 2 text", "Document 3 text"]  # Replace with actual documents
document_embeddings = encode_documents(documents)

# Convert to FAISS index format
embedding_dim = document_embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)
faiss.normalize_L2(document_embeddings.numpy())
index.add(document_embeddings.numpy())

# Save the index for future use
faiss.write_index(index, 'gtr_index.index')
```

### Storage Configurations

Store the **GTR** index on different storage configurations:

#### NVMe on EC2

```bash
sudo mkfs -t ext4 /dev/nvme0n1
sudo mount /dev/nvme0n1 /mnt/nvme
mv gtr_index.index /mnt/nvme/
```

#### EBS

```bash
sudo mkfs -t ext4 /dev/xvdf
sudo mount /dev/xvdf /mnt/ebs
mv gtr_index.index /mnt/ebs/
```

#### DRAM

For in-memory storage, load the index directly into memory:

```python
index = faiss.read_index('gtr_index.index')
```

## Model Deployment

### Deploying LLaMA2 34B Model

The **LLaMA2 34B** model is deployed on **Amazon SageMaker** for answer generation. Given its large size, we use a powerful instance such as **ml.p4d.24xlarge** for inference.

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

For each BioASQ query, the following steps are executed:

1. **Embed the Query** using the GTR model to generate a dense vector representation.
2. **Retrieve the Top-k Documents** using Faiss from the GTR index.
3. **Generate the Answer** using the LLaMA2 34B model based on the retrieved documents.

```python
query = "What are the effects of aspirin on heart disease?"
query_tokens = tokenizer(query, return_tensors="pt")

# Embed the query
query_embedding = model(**query_tokens).pooler_output
faiss.normalize_L2(query_embedding.numpy())

# Retrieve the top k documents
k = 5
distances, indices = index.search(query_embedding.numpy(), k)
retrieved_docs = [documents[i] for i in indices[0]]

# Generate the answer using LLaMA2
response = predictor.predict({"input": query, "context": retrieved_docs})
```

### Profiling and Performance Metrics

We measure **latency**, **memory usage**, and **throughput** for each phase (query embedding, retrieval, augmentation, and generation).

```python
import time

# Record the start time for query embedding
start = time.time()
query_tokens = tokenizer(query, return_tensors="pt")
embedding_time = time.time() - start

# Record retrieval time
start = time.time()
distances, indices = index.search(query_embedding.numpy(), k)
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

After processing all BioASQ queries, the system will print the following performance statistics for each query:

1. **Query Embedding Latency**
2. **Document Retrieval Latency**
3. **Answer Generation Latency**
4. **Total Latency** (query to answer)
5. **Memory Utilization** during embedding, retrieval, and generation.
6. **Throughput** (queries per second).

Example output:

```plaintext
Query: What are the effects of aspirin on heart disease?
Query Embedding Latency: 0.35s
Document Retrieval Latency: 0.25s
Generation Latency: 4.0s
Total Latency: 4.6s
Memory Utilization: 12GB
Throughput: 0.6 queries/sec
```



