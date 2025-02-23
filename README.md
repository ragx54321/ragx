# RAGX

*A repository of benchmarks for retrieval augmented generation..*

---

## Overview

Retrieval-Augmented Generation (RAG) is an advanced architecture that combines information retrieval techniques with large language models (LLMs) to improve the quality and relevance of generated responses. The RAG pipeline typically consists of three main phases: **Search and Retrieval**, **Augmentation**, and **Referenced Generation**.

1. **Search and Retrieval**: In this phase, a query is used to retrieve relevant documents or passages from a large corpus of data. The search can be conducted using either **keyword-based** or **embedding-based** retrieval methods. Keyword-based retrieval (e.g., BM25) relies on token matches between the query and the documents, while embedding-based methods (e.g., ColBERT, Doc2Vec, GTR) generate vector representations of documents and queries, which are then used for semantic search.

2. **Augmentation**: Once relevant passages are retrieved, the query is augmented by incorporating these passages. This augmented query, enriched with additional information, is then passed to the LLM.

3. **Referenced Generation**: The final stage involves using the augmented query as input to an LLM (e.g., LLAMA2, GPT, Claude) to generate a response, leveraging the information from the retrieved documents.

This benchmark suite aims to evaluate and compare various configurations of RAG systems across these phases. Users can adjust parameters like the type of retriever, top-k values, and LLM backends to assess their impact on retrieval and generation performance.

The dataset we used to conduct our analysis was PubMed, along with a set of medical questions (bioasq). The information regarding these datasets can be found in the following github - https://github.com/neulab/ragged/tree/main/. It is important to note, that the user can select the dataset they wish to use.

This benchmarks are based off of prior work - https://github.com/neulab/ragged.

---

## Table of Contents

- [Overview](#overview)
- [Benchmarks](#benchmarks)
- [Installation](#installation)
- [AWS Deployment](#aws-deployment)
- [Directory Structure](#directory-structure)

---

## Benchmarks

This section details the various benchmark models and experiments. Each benchmark includes its own python scripts.

Each benchmark's search and retrieval phase can be described as keyword or embedding based. Keyword based retrieval focuses on using the tokens within a query to determine the most relevant documents. Embedding based retrieval casts the initial query into an embedding which is then used to search through a database of embedded documents. 

- **Benchmark 1: BM25-LLAMA2 (keyword based)**
  - *Description:* BM25 is a ranking function used in search engines to determine how relevant a document is to a search query by weighing the importance of each term within the document. BM25 determines the most relevant documents by considering both how often query terms appear and how unique they are across all documents.
  - *Key Features:* The structure used to hold the posting lists of BM25 is an inverted index. To generate an inverted index, please refer to RAGGED's open source code - https://github.com/neulab/ragged/tree/main/retriever/BM25. In addition, pyserini provides clear documentation - https://github.com/castorini/pyserini. The script you will find in our repository (under Benchmarks/src_retrievers/BM25) is used to search the generated inverted index with a set of queries. 
  
- **Benchmark 2: SPLADEv2-LLAMA2 (keyword/embedding based)**
  - *Description:* SPLADEv2 uses a transformer encoder to generate sparse, term-weighted representations by expanding queries and documents with related vocabulary. These sparse vectors directly map to an inverted index, where non-zero weights indicate which documents contain specific terms. 
  - *Key Features:* SPLADEv2 uses an inverted index to hold its posting lists. SPLADEv2 has its own github from the authors of algorithm - https://github.com/naver/splade. Please refer to their github for benchmarking (both searching an inverted index and generating an inverted index). The authors of splade use lucene as the backend for their inverted index. Lucene is also the backed for pyserini which was used for BM25. 

- **Benchmark 3: ColBERT-LLAMA2 (embedding based)**
  - *Description:* ColBERT’s embedding model is based on a transformer architecture (BERT Base) that converts each token in a query or document into a contextualized embedding. 
  - *Key Features:* To host ANN search, we use ColBERT to generate embeddings and store these embeddings in an HNSW (https://arxiv.org/pdf/1603.09320). HNSWs provide high quality retrieval at fast speeds. However, they can consume a large memory footprint. We make use of faiss's implementation of HNSW - https://github.com/facebookresearch/faiss/blob/main/faiss/impl/HNSW.cpp. 
 
- **Benchmark 4: Doc2Vec-LLAMA2 (embedding based)**
  - *Description:* Doc2Vec is an extension of Word2Vec that learns fixed-length vector representations for variable-length texts, such as sentences, paragraphs, or entire documents.  These document embeddings effectively capture semantic similarities, enabling tasks like document clustering, classification, and similarity search.
  - *Key Features:* To host ANN search, we use Doc2Vec to generate embeddings, and the embeddings are stored in an HNSW. HNSWs provide high quality retrieval at fast speeds. However, they can consume a large memory footprint. We make use of faiss's implementation of HNSW - https://github.com/facebookresearch/faiss/blob/main/faiss/impl/HNSW.cpp. 

- **Benchmark 5: GTR-LLAMA2 (embedding based)**
  - *Description:* GTR is a dense embedding model built on the T5 transformer architecture, specifically fine-tuned for retrieval tasks. It generates semantically vector representations for queries and documents using contrastive learning, ensuring that related pairs are close in the embedding space. You can access Google's source code for GTR here - https://github.com/google-research/t5x_retrieval. 
  - *Key Features:* To host ANN search, we use GTR to generate embeddings, and the embeddings are stored in an HNSW.

For all embedding based retrieval models, we have provided corresponding scripts to generate the embeddings of a corpus of passages. The scripts will also store the generated embeddings in an HNSW. Please be mindful that depending on your document dataset size, embedding can be an extremely time consuming process. Embedding 50 million passages with GTR on a 4 A10 GPUs took over 3 days. 

For keyword based retrieval models, we have pointed to alternative githubs, which provide detailed and thorough instructions on how to generate the required inverted indices. Please refer to these. 

Our github also provides the needed scripts to actually query the HNSW/Inverted Index to search for relevant documents. The output of these scripts will be latency. 

To understand the inner-workings of HNSWs or Inverted Indicies, the corresponding backend code for these structures needs to be instrumented. To do so for HNSWs, clone FAISS's github, and instrument the desired portions of the HNSW implementation. Once the code is instrumented, you must locally build it. To instrument pyserini or lucene's implementation of an inverted index is quite tricky since there is a number of backend java files. Luckily, pyserini provides a number of useful functions to illuminate the inner workings of an inverted index. Moreover, SPLADE's github provides opportunities for intstrumentation. 

Below, we have included the runtimes for embedding, index construction as well as their associated costs (for embeddings we used 4 A10 GPUs with the cost based on EC2 pricing and for building an HNSW we used an ec2 instance with 756 GB of memory). We used these instances due to the needed compute and DRAM requirements of the index build process, **which typically uses substantially more memory than the size of the raw text or embeddings**.

|                                         | 0.5 Million Passages | 5 Million Passages | 50 Million Passages | 500 Million Passages |
|------------------------------------------|-------------------|-------------------|--------------------|--------------------|
| BM25 - Tokenization Time                  | < 1 minute (< $1)  | < 1 minute (< $1) | < 1 minute (< $1)  | TBD                 |
| BM25 - Inverted Index Construction Time   | < 10 minutes (< $1)| < 30 minutes (< $3)| < 1 hour (< $6)    | < 4 hours (<$24)   |
| SPLADEv2 - Embedding Time                 | 2 Hours ($12)      | 16 Hours ($96)    | 72 hours ($432)    | TBD                 |
| SPLADEv2 - Inverted Index Construction Time| < 30 minutes (< $3)| < 30 minutes (< $3)| < 1 hour (< $6)   | < 4 hours (<$24)   |
| ColBERT - Embedding Time                  | < 1 hour (< $6)    | 6 hours ($36)     | 36 hours ($216)    | TBD                 |
| ColBERT - HNSW Construction Time          | < 15 minutes (< $1.5)| < 1 hour (< $6) | 3 hours ($18)      | 18 hours ( $108)    |
| Doc2Vec - Embedding Time                  | < 30 minutes (< $3)| 2 hours ($12)     | 12 hours ($72)     | TBD                 |
| Doc2Vec - HNSW Construction Time          | < 15 minutes (< $1.5)| < 1 hour (< $6) | 4 hours ($24)      | 20 hours ($120)     |
| GTR - Embedding Time                      | < 1 hour (< $6)    | 16 hours (96)     | 72 hours ($432)    | TBD                 |
| GTR - HNSW Construction Time              | < 30 minutes (< $3)| < 1 hour (< $6)   | 6 hours ($36)      | 24 hours ($144)     |

Below, we have included the size of the raw embeddings needed for all the different datasets. If you plan on generating an HNSW, please be aware that you need **2-3x of the dataset size in GBs available in DRAM**. For example to build an HNSW with ColBERT embeddings of 500 million passages, you will need 768GB of DRAM.
|                                         | 0.5 Million Passages | 5 Million Passages | 50 Million Passages | 500 Million Passages |
|-----------------------------------------|----------------------|--------------------|---------------------|----------------------|
| BM25 - Dataset Size (Posting Lists)     | .26 GB               | 3.2 GB             | 32 GB               |          320GB       |
| SPLADEv2 - Dataset Size (Posting Lists) | .5 GB                | 2 GB               | 18 GB               |          180GB       |
| ColBERT - Dataset Size (Embeddings)     | 0.256                | 2.56 GB            | 25.6 GB             |          256 GB      |
| Doc2Vec - Dataset Size (Embeddings)     | 0.6 GB               | 6 GB               | 60 GB               |          600 GB      |
| GTR - Dataset Size (Embeddings)         | 1.5 GB               | 15 GB              | 150 GB              |          1500 GB     |

The following section will detail how to make use of the custom scripts we have developed to benchmark RAGs. 

### Retrieval Configurations

- **Top-k Retrieval**: Controls the number of top documents retrieved during the search phase.
- **Embedding Generation**: Each retrieval method has scripts for generating embeddings of large corpora, which are then stored in a vector database for fast retrieval.
- **Vector Databases**: We support different vector databases like FAISS for HNSW and IVF, and ScANN for scalable nearest neighbor search.

To enable storing the vector databases (representations such as embeddings or posting lists) on NVMe, we need to build FAISS locally. Using the following code for it. ALternatively, to store the vector database on DRAM, ignore the following. 


```bash
export MKLROOT=~/miniconda3/envs/RAG/lib
cmake -B build -DBUILD_TESTING=ON -DFAISS_ENABLE_GPU=OFF \
   -DFAISS_OPT_LEVEL=avx2 \
   -DFAISS_ENABLE_C_API=ON \
   -DCMAKE_BUILD_TYPE=Release \
   -DBLA_VENDOR=Intel10_64_dyn .
make -C build -j10 swigfaiss
(cd build/faiss/python ; python3 setup.py build)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/build/lib
export PYTHONPATH=$(pwd)/build/faiss/python/build/lib
(cd tests ; OMP_NUM_THREADS=1 python -m unittest discover -v)
```

### LLM Backends

For referenced generation, the retrieved passages are passed to an LLM. In this benchmark suite, we focus on **LLAMA2** as the backend for referenced generation, but users can configure other LLMs (e.g., GPT, Claude) depending on their preferences.

---

## Datasets

We provide two core datasets for evaluation:

1. **PubMed**: A comprehensive dataset of biomedical articles, useful for benchmarking RAG systems in the medical domain. The data can be downloaded and processed for retrieval and question-answering tasks. For further information on the dataset, refer to the PubMed website.

2. **BioASQ**: A medical question dataset containing queries designed for answering tasks, particularly useful for evaluating RAG-based systems on question-answering tasks. The dataset can be used to test the performance of different retrieval and generation configurations.

Additionally, the repository includes scripts to download other popular datasets like **Wikipedia** and other public text corpora for broader benchmarking.

For more information and steps to download the datasets, please refer RAGGED paper, which we used [https://github.com/neulab/ragged](https://github.com/neulab/ragged)

## Installation


There are two ways to run the benchmarks: local premise or on a public cloud such as AWS. We provide both steps.

```bash
# Clone the repository
git clone https://github.com/yourusername/RAGX.git
cd RAGX

# Install dependencies, please be mindful of conflicting requirements with SPLADEv2's github
pip install -r requirements.txt

```

## AWS
To setup RAG on AWS, we stuck strictly to the public examples provided by AWS. We made use of the available instances on AWS designed specifically for RAG applications. Below, we have included examples we based our methodology and experimental setup on.

- https://aws.amazon.com/blogs/big-data/build-scalable-and-serverless-rag-workflows-with-a-vector-engine-for-amazon-opensearch-serverless-and-amazon-bedrock-claude-models/
- https://aws.amazon.com/blogs/big-data/integrate-sparse-and-dense-vectors-to-enhance-knowledge-retrieval-in-rag-using-amazon-opensearch-service/
- https://github.com/aws-samples/rag-with-amazon-opensearch-and-sagemaker
- https://github.com/aws-samples/rag-with-amazon-bedrock-and-opensearch
- https://github.com/aws-samples/serverless-rag-demo
- https://github.com/aws-samples/rag-chatbot-with-bedrock-opensearch-and-document-summaries-in-govcloud


To deploy on AWS, please check the README in aws/ directory.

---



## Directory Structure

- **`retrievers/`**: Code for implementing retrieval models like BM25, SPLADEv2, ColBERT, Doc2Vec, GTR.
- **`llm_backend/`**: Integrations for LLMs such as LLAMA2, GPT, Claude to generate responses.


The readme and comments in the code in this repo have been enhanced with open-sourced GenAI tools.


## References

- RAGGED paper: [https://github.com/neulab/ragged](https://github.com/neulab/ragged)
- BM25: Robertson, S., & Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond.
- SPLADEv2: Humeau, S., Shuster, K., & West, P. (2021). SPLADE: Sparse lexical and dense retrieval.
- ColBERT: Khattab, O., & Zaharia, M. (2020). ColBERT: Efficient and effective passage search via contextualized late interaction over BERT.
- Doc2Vec: Le, Q. V., & Mikolov, T. (2014). Distributed representations of sentences and documents.
- GTR: (2020). GTR: Retrieval-based transformer.


## License

This repository is licensed under the MIT License. See [LICENSE](LICENSE) for more details.

