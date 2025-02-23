### 1. **Retrieval Models**

The retrieval step is crucial in RAG for selecting relevant documents that will be used for generating answers. The repository includes the following retrieval models:

- **BM25**: Uses **Inverted Index** and **Postings List** for ranking documents based on term frequency and inverse document frequency (TF-IDF). Ideal for systems focusing on exact term matching.
- **Doc2Vec**: Embeds documents into fixed-length vectors, which are used for similarity-based document retrieval. This method is well-suited for semantic matching.
- **SPLADE**: Utilizes **sparse transformers** for efficient document retrieval, focusing on sparse representations. Suitable for handling large-scale document collections.
- **GTR**: A transformer-based retrieval model that computes contextualized representations for documents and queries, enhancing the semantic matching.

### 2. **Generative Model**

Once the documents are retrieved, a **generative model** is used to formulate an answer. The repository uses:

- **LLaMA2 34B**: A large-scale transformer model that generates natural language responses based on the retrieved context. You can easily swap this model for other variants like LLaMA2 70B for higher accuracy at the cost of computational resources.


---

## How to Customize and Perform Sensitivity Analysis

The system is designed to be flexible, allowing users to easily experiment with different configurations. Here’s how you can change and test various components:

### 1. **Changing the Retrieval Model**

You can switch between different retrieval models by modifying the readme related to retrieval (e.g., `BM25-LLaMA`).

#### For BM25:
- BM25 uses an inverted index to rank documents by their term relevance.
- Update the retrieval code in the respective readme:
  ```python
  from rank_bm25 import BM25Okapi
  bm25 = BM25Okapi(tokenized_docs)
  ```
  You can adjust parameters like **Top-k** (how many documents to retrieve).

#### For Doc2Vec:
- Use Doc2Vec embeddings to represent documents as vectors for similarity-based retrieval.
- Modify the readme to use **Faiss** or other similarity search methods:
  ```python
  from gensim.models import Doc2Vec
  model = Doc2Vec.load("doc2vec_model")
  vector = model.infer_vector(query)
  ```
  
#### For SPLADE and GTR:
- Use the **SPLADE** or **GTR** model from the transformers library:
  ```python
  from transformers import AutoTokenizer, AutoModelForSequenceClassification
  tokenizer = AutoTokenizer.from_pretrained("splade/splade-v2")
  model = AutoModelForSequenceClassification.from_pretrained("splade/splade-v2")
  ```

Each retrieval model can be evaluated for its effect on system performance using the **performance_evaluation.ipynb** readme.

### 2. **Changing the Generative Model**

To switch the generative model (e.g., from **LLaMA2 34B** to **LLaMA2 70B** or another transformer), modify the relevant readme (`rag_pipeline.ipynb` or any generation-related readme):

#### LLaMA2 Model Change:
- Modify the model initialization code to use a different variant of **LLaMA2**:
  ```python
  from transformers import LlamaForCausalLM
  model = LlamaForCausalLM.from_pretrained("meta/llama-2-70b")
  ```

#### Hardware Considerations:
- Be mindful of the hardware requirements for larger models like **LLaMA2 70B**. Ensure that you are using a **multi-GPU setup** or the necessary **memory optimizations**.

### 3. **Optimizing Attention Mechanisms**

For faster attention mechanisms, you can switch to using **FlashAttention**:
- Install **FlashAttention**:
  ```bash
  pip install flash-attn
  ```
- Modify the model initialization to enable **FlashAttention**:
  ```python
  model = LlamaForCausalLM.from_pretrained("meta/llama-2-34b", flash_attention=True)
  ```

This can significantly reduce memory usage and speed up computations, especially with larger models.

### 4. **Adjusting Top-K Retrieval**

To change the number of top documents retrieved during the retrieval step (which can impact both speed and accuracy), modify the **Top-k** parameter in the retrieval code:
```python
top_k = 10  # Adjust this value to retrieve more or fewer documents
distances, indices = index.search(query_embedding, top_k)
```

### 5. **Performing Sensitivity Analysis**

To evaluate the impact of different configurations on the system's performance:
- Modify parameters like **model size** (LLaMA2 34B vs 70B), **top-k**, and **embedding model**.
- Use the **performance_evaluation.ipynb** to profile latency, memory usage, and throughput for various configurations.
  
**Example Sensitivity Experiment**:
- Change the retrieval model from **BM25** to **Doc2Vec** and record the performance metrics.
- Switch from **LLaMA2 34B** to **LLaMA2 70B** and compare the changes in latency and memory usage.
- Use **FlashAttention** with **LLaMA2 34B** and measure its impact on throughput and latency.


