## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Dataset Preparation](#dataset-preparation)
   - [BioASQ Dataset](#bioasq-dataset)
4. [Environment Setup](#environment-setup)
5. [Indexing and Retrieval](#indexing-and-retrieval)
   - [Doc2Vec Index](#doc2vec-index)
   - [Storage Configurations](#storage-configurations)
6. [Model Deployment](#model-deployment)
   - [Deploying LLaMA2 34B Model](#deploying-llama2-34b-model)
7. [Pipeline Execution](#pipeline-execution)
   - [Processing BioASQ Queries](#processing-bioasq-queries)
   - [Profiling and Performance Metrics](#profiling-and-performance-metrics)
8. [Performance Statistics](#performance-statistics)
9. [Conclusion](#conclusion)

## Overview

**Doc2Vec** is an extension of the **Word2Vec** model that learns fixed-length vector representations for entire documents, rather than just words. It maps variable-length text to a fixed-size vector space, capturing semantic information from the entire document. By learning to predict a document's context based on its content, **Doc2Vec** generates embeddings that can be used for similarity searches. It is particularly useful in retrieval tasks where the similarity between entire documents and queries needs to be computed efficiently. This makes **Doc2Vec** a powerful choice for document retrieval in question answering systems.

## Prerequisites

- AWS account with permissions for SageMaker, EC2, and S3 services.
- Python 3.x installed.
- AWS CLI and Boto3 configured.
- Basic knowledge of machine learning, Doc2Vec, and cloud services.

### Install Required Libraries

```bash
pip install transformers datasets faiss-cpu gensim boto3
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
   pip install transformers datasets faiss-cpu gensim boto3
   ```

## Indexing and Retrieval

### Doc2Vec Index

**Doc2Vec** creates vector embeddings for documents by training a model that learns fixed-length representations for each document. These embeddings can be indexed and used for efficient retrieval with **Faiss**.

1. **Preprocessing the Documents**: Tokenize the documents and create embeddings using **Doc2Vec**.
2. **Index the Documents**: Use **Faiss** to index the dense document embeddings for efficient retrieval.

```python
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import faiss
import numpy as np

# Load documents
documents = ["Document 1 text", "Document 2 text", "Document 3 text"]  # Replace with actual documents

# Preprocess and tag documents
tagged_data = [TaggedDocument(words=doc.split(), tags=[str(i)]) for i, doc in enumerate(documents)]

# Train a Doc2Vec model
model = Doc2Vec(vector_size=300, window=2, min_count=1, workers=4)
model.build_vocab(tagged_data)
model.train(tagged_data, total_examples=model.corpus_count, epochs=10)

# Create embeddings for the documents
document_embeddings = np.array([model.infer_vector(doc.split()) for doc in documents])

# Normalize the embeddings
faiss.normalize_L2(document_embeddings)

# Create a FAISS index
embedding_dim = document_embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)
index.add(document_embeddings)

# Save the index for future use
faiss.write_index(index, 'doc2vec_index.index')
```

### Storage Configurations

Store the **Doc2Vec** index on different storage configurations:

#### NVMe on EC2

```bash
sudo mkfs -t ext4 /dev/nvme0n1
sudo mount /dev/nvme0n1 /mnt/nvme
mv doc2vec_index.index /mnt/nvme/
```

#### EBS

```bash
sudo mkfs -t ext4 /dev/xvdf
sudo mount /dev/xvdf /mnt/ebs
mv doc2vec_index.index /mnt/ebs/
```

#### DRAM

For in-memory storage, load the index directly into memory:

```python
index = faiss.read_index('doc2vec_index.index')
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

1. **Embed the Query** using the Doc2Vec model to generate a dense vector representation.
2. **Retrieve the Top-k Documents** using Faiss from the Doc2Vec index.
3. **Generate the Answer** using the LLaMA2 34B model based on the retrieved documents.

```python
query = "What are the effects of aspirin on heart disease?"
query_tokens = query.split()  # Tokenize query

# Embed the query
query_embedding = model.infer_vector(query_tokens)
faiss.normalize_L2(query_embedding)

# Retrieve the top k documents
k = 5
distances, indices = index.search(np.array([query_embedding]), k)
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
query_embedding = model.infer_vector(query.split())
embedding_time = time.time() - start

# Record retrieval time
start = time.time()
distances, indices = index.search(np.array([query_embedding]), k)
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