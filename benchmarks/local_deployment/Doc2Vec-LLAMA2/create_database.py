import os
import time
import ujson as json
import numpy as np
from tqdm import tqdm
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.parsing.preprocessing import preprocess_string
import faiss  # For building the HNSW index

def load_and_process_jsonl(file_path, content_field, max_docs=None):
    """Load and preprocess data from a JSONL file into a list of TaggedDocuments."""
    documents = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for i, line in enumerate(tqdm(file, desc=f"Loading {os.path.basename(file_path)}", total=max_docs or sum(1 for _ in open(file_path)))):
            if max_docs is not None and i >= max_docs:
                break  # Stop reading if max_docs is reached
            item = json.loads(line)
            content = item.get(content_field, '').strip()
            if content:
                words = preprocess_string(content)
                if words:
                    documents.append(TaggedDocument(words=words, tags=[item['id']]))

    return documents

def train_doc2vec(corpus, vector_size=300, epochs=10, num_threads=1, model_path="doc2vec_model"):
    """Train a Doc2Vec model on the given corpus and ensure vector size is 300."""
    print(f"Training Doc2Vec model using {num_threads} threads with vector size {vector_size}...")
    
    model = Doc2Vec(vector_size=vector_size, window=2, min_count=1, workers=num_threads, negative=5)  # Include negative sampling
    
    # Build vocabulary
    model.build_vocab(corpus)

    # Train model
    for epoch in range(epochs):
        model.train(corpus, total_examples=len(corpus), epochs=1)
        model.alpha -= 0.002  # Decrease the learning rate
        model.min_alpha = model.alpha  # Fix the learning rate, no decay
        print(f"Completed epoch {epoch + 1}/{epochs}")

    # Save the model (this will generate the .npy files)
    model.save(model_path)

    # Verify the vector size for the first 100 embeddings
    for i in range(min(100, len(corpus))):
        vector = model.dv[i]
        if len(vector) != vector_size:
            raise ValueError(f"Vector size mismatch for document {i}. Expected {vector_size}, got {len(vector)}.")
    
    print("First 100 document vectors have the correct size of 300.")
    return model

def create_faiss_hnsw_index(doc_vectors, hnsw_path, M=32, ef_construction=200, num_threads=4):
    """Create a FAISS HNSW index for the document vectors."""
    print("Creating FAISS HNSW index...")
    dimension = doc_vectors.shape[1]
    index = faiss.IndexHNSWFlat(dimension, M)
    index.hnsw.efConstruction = ef_construction
    faiss.omp_set_num_threads(num_threads)  # Set the number of threads for FAISS
    index.add(doc_vectors)
    print("FAISS index created successfully.")

    # Save the index to a file
    faiss.write_index(index, hnsw_path)

    return index

if __name__ == "__main__":
    corpus_file = 'PASSAGE CORPUS PATH'
    model_path = "MODEL PATH"
    hnsw_path = "HNSW PATH"

    # Load and preprocess the corpus
    corpus = load_and_process_jsonl(corpus_file, 'contents', max_docs=10000000)
    print(f"{len(corpus)} documents loaded from the corpus.")

    # Train the Doc2Vec model and verify vector sizes for the first 100 vectors
    model = train_doc2vec(corpus, vector_size=300, num_threads=16, model_path=model_path)

    # Extract document vectors from the model
    doc_vectors = np.array([model.dv[i] for i in range(len(corpus))])

    # Create and save the FAISS HNSW index
    create_faiss_hnsw_index(doc_vectors, hnsw_path, M=32, ef_construction=200, num_threads=16)