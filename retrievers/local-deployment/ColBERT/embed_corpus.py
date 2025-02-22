import os
import json
import numpy as np
import faiss
import torch
from torch.nn import DataParallel
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
import time
import h5py

# Limit the script to run on only 3 GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

# Function to load data from jsonl file with a progress bar
def load_jsonl(file_path, num_docs=None, key='text'):
    print(f"Loading data from {file_path}...")
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for i, line in tqdm(enumerate(file), desc="Loading JSONL", unit="lines"):
            if num_docs is not None and i >= num_docs:
                break
            item = json.loads(line)
            data.append(item.get(key, ''))  # Adjust based on your JSON structure
    print(f"Loaded {len(data)} entries from the JSONL file.")
    return data

# Function to generate embeddings using BERT model
def generate_embeddings(texts, model, tokenizer, device):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling over tokens
    return embeddings.cpu().numpy()

# Function to evaluate the index
def evaluate(index, xq, k, nq, gt=None):
    t0 = time.time()
    D, I = index.search(xq, k)
    t1 = time.time()
    missing_rate = (I == -1).sum() / float(k * nq)
    recall_at_1 = (I == gt[:, :1]).sum() / float(nq) if gt is not None else 0.0
    print("\t %7.3f ms per query, R@1 %.4f, missing rate %.4f" % (
        (t1 - t0) * 1000.0 / nq, recall_at_1, missing_rate))

# Function to save embeddings as HDF5
def save_as_hdf5(file_path, embeddings, dataset_name="train"):
    print(f"Saving embeddings as HDF5 to {file_path}...")
    with h5py.File(file_path, 'w') as hf:
        hf.create_dataset(dataset_name, data=embeddings)
    print(f"Saved embeddings to {file_path}")

# Main function to create FAISS indices or save embeddings in HDF5 and run evaluations
def main(corpus_file, query_file, num_docs=100000, num_queries=1000, k=10, index_save_file="", save_hdf5=False, hdf5_file=""):
    # Set up the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load BERT model and tokenizer
    model_name = 'bert-base-uncased'
    model_name_custom = 'INSERT ColBERT MODEL PATH'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name_custom)

    # Move model to the appropriate device before wrapping it in DataParallel
    model = model.to(device)

    # Wrap the model with DataParallel if more than one GPU is available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = DataParallel(model, device_ids=[0, 1, 2])  # Use the 3 GPUs you selected

    model.eval()

    # Load queries
    query_data = load_jsonl(query_file, num_docs=num_queries, key='question')

    # Generate embeddings for queries
    print("Generating embeddings for queries...")
    query_embeddings = []
    batch_size = 64
    for i in tqdm(range(0, len(query_data), batch_size), desc="Embedding Queries", unit="batch"):
        batch_texts = query_data[i:i+batch_size]
        embeddings = generate_embeddings(batch_texts, model, tokenizer, device)
        query_embeddings.append(embeddings)
    query_embeddings = np.vstack(query_embeddings).astype('float32')

    # Load or create FAISS index, or save embeddings as HDF5
    if save_hdf5:
        # Save document embeddings as HDF5
        print("Generating embeddings for documents to save as HDF5...")
        corpus_data = load_jsonl(corpus_file, num_docs=num_docs, key='contents')

        corpus_embeddings = []
        for i in tqdm(range(0, len(corpus_data), batch_size), desc="Embedding Corpus", unit="batch"):
            batch_texts = corpus_data[i:i+batch_size]
            embeddings = generate_embeddings(batch_texts, model, tokenizer, device)
            corpus_embeddings.append(embeddings)
        corpus_embeddings = np.vstack(corpus_embeddings).astype('float32')

        # Save the embeddings as HDF5
        save_as_hdf5(hdf5_file, corpus_embeddings)
    
    else:
        # Create FAISS index or load an existing one
        if os.path.exists(index_save_file):
            print(f"Loading existing index from '{index_save_file}'")
            index = faiss.read_index(index_save_file)
        else:
            print("Generating new index since none was found.")
            # Load documents
            corpus_data = load_jsonl(corpus_file, num_docs=num_docs, key='contents')

            # Generate embeddings for documents
            print("Generating embeddings for documents...")
            corpus_embeddings = []
            for i in tqdm(range(0, len(corpus_data), batch_size), desc="Embedding Corpus", unit="batch"):
                batch_texts = corpus_data[i:i+batch_size]
                embeddings = generate_embeddings(batch_texts, model, tokenizer, device)
                corpus_embeddings.append(embeddings)
            corpus_embeddings = np.vstack(corpus_embeddings).astype('float32')

            # Create HNSW Flat index
            print("Creating HNSW Flat index")
            d = corpus_embeddings.shape[1]
            index = faiss.IndexHNSWFlat(d, 32)
            index.hnsw.efConstruction = 80
            index.verbose = True
            index.add(corpus_embeddings)

            # Save the index for future use
            faiss.write_index(index, index_save_file)
            print(f"HNSW Flat index saved at '{index_save_file}'.")

        # Set up FAISS evaluation
        d = query_embeddings.shape[1]
        nq = query_embeddings.shape[0]

        # Evaluation using the loaded index
        if 'hnsw' in todo:
            print("Evaluating HNSW Flat index")
            for efSearch in [16, 32, 64, 128, 256]:
                for bounded_queue in [True, False]:
                    print("efSearch", efSearch, "bounded queue", bounded_queue, end=' ')
                    index.hnsw.search_bounded_queue = bounded_queue
                    index.hnsw.efSearch = efSearch
                    evaluate(index, query_embeddings, k, nq)

if __name__ == "__main__":
    todo = ['hnsw']  # Define which tests to run
    main(
        corpus_file='CORPUS PATH',
        query_file='QUERY PATH',
        num_docs=500000,
        num_queries=1,
        k=30,
        index_save_file="SAVE INDEX PATH",  # Specify the file name for the index
        save_hdf5=False,  # Set to True to save embeddings as HDF5, False to save as FAISS index
        hdf5_file="SAVE EMBEDDINGS PATH"  # File to save HDF5 embeddings
    )
