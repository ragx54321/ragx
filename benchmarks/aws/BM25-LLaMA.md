
## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Dataset Preparation](#dataset-preparation)
   - [BioASQ Dataset](#bioasq-dataset)
4. [Environment Setup](#environment-setup)
5. [Indexing and Retrieval](#indexing-and-retrieval)
   - [BM25 Index](#bm25-index)
   - [Postings List and Inverted Index](#postings-list-and-inverted-index)
   - [Storage Configurations](#storage-configurations)
6. [Model Deployment](#model-deployment)
   - [Deploying LLaMA2 34B Model](#deploying-llama2-34b-model)
7. [Pipeline Execution](#pipeline-execution)
   - [Processing BioASQ Queries](#processing-bioasq-queries)
   - [Profiling and Performance Metrics](#profiling-and-performance-metrics)
8. [Performance Statistics](#performance-statistics)
9. [Conclusion](#conclusion)

## Overview

## Prerequisites

- AWS account with permissions for SageMaker, EC2, and S3 services.
- Python 3.x installed.
- AWS CLI and Boto3 configured.
- Basic knowledge of machine learning, BM25, and cloud services.

### Install Required Libraries

```bash
pip install gensim faiss-cpu transformers datasets boto3 rank_bm25
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
   pip install gensim faiss-cpu transformers datasets boto3 rank_bm25
   ```

## Indexing and Retrieval

### BM25 Index

The **BM25** algorithm relies on an **inverted index** where each term points to a list of documents (called the postings list) in which the term appears. This inverted index allows for fast document retrieval based on term frequency and inverse document frequency.

1. **Tokenize the documents**: Preprocess and tokenize the text data for indexing.
2. **Build the inverted index** using the **BM25** implementation from the **rank_bm25** library.

```python
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize

# Tokenize the documents
tokenized_data = [word_tokenize(doc.lower()) for doc in text_data]

# Create the BM25 object and index the documents
bm25 = BM25Okapi(tokenized_data)

# Save the BM25 object for future use
import pickle
with open('bm25_index.pkl', 'wb') as f:
    pickle.dump(bm25, f)
```

### Postings List and Inverted Index

The inverted index is created by breaking down each document into tokens and then mapping each token to the document IDs where it appears. This allows the BM25 algorithm to quickly compute the relevance of documents given a query.

Example code to construct the inverted index (for understanding):

```python
from collections import defaultdict

# Create the inverted index
inverted_index = defaultdict(list)

for doc_id, doc in enumerate(text_data):
    tokens = word_tokenize(doc.lower())
    for token in set(tokens):
        inverted_index[token].append(doc_id)

# Save the inverted index
import pickle
with open('inverted_index.pkl', 'wb') as f:
    pickle.dump(inverted_index, f)
```

### Storage Configurations

Store the **BM25** index on different storage configurations:

#### NVMe on EC2

```bash
sudo mkfs -t ext4 /dev/nvme0n1
sudo mount /dev/nvme0n1 /mnt/nvme
mv bm25_index.pkl /mnt/nvme/
```

#### EBS

```bash
sudo mkfs -t ext4 /dev/xvdf
sudo mount /dev/xvdf /mnt/ebs
mv bm25_index.pkl /mnt/ebs/
```

#### DRAM

For in-memory storage, load the index directly into memory:

```python
with open('bm25_index.pkl', 'rb') as f:
    bm25 = pickle.load(f)
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

1. **Embed the Query** using Doc2Vec or similar embeddings.
2. **Retrieve the Top-k Documents** using BM25 from the inverted index.
3. **Generate the Answer** using the LLaMA2 34B model based on the retrieved documents.

```python
query = "What are the effects of aspirin on heart disease?"
query_tokens = word_tokenize(query.lower())

# Perform retrieval using BM25
scores = bm25.get_scores(query_tokens)
top_k_docs = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:5]
retrieved_docs = [text_data[i] for i, _ in top_k_docs]

# Generate the answer using LLaMA2
response = predictor.predict({"input": query, "context": retrieved_docs})
```

### Profiling and Performance Metrics

We measure **latency**, **memory usage**, and **throughput** for each phase (query embedding, retrieval, augmentation, and generation).

```python
import time

# Record the start time for query embedding
start = time.time()
query_tokens = word_tokenize(query.lower())
embedding_time = time.time() - start

# Record retrieval time
start = time.time()
scores = bm25.get_scores(query_tokens)
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
