This directory contains the benchmarking code and utilities for evaluating the **Retrieval-Augmented Generation (RAG)** system. The benchmarks measure the latency of different stages in the pipeline, including retrieval, passage fetching, tokenization, and inference.

## Directory Structure

- `local_deployment/`: Code and setup for running the RAG pipeline locally.
- `aws/`: Code and setup for running the RAG pipeline on AWS.

## Setup

### Local Deployment

For **local deployment**, follow these steps:

#### 1. Install Dependencies

Start by installing the required Python packages. Navigate to the root of the repository and run:

```bash
# Embed Corpus (if necessary)
cd retrievers
cd [desired_benchmark]  # Replace with the desired benchmark (e.g., ColBERT)
vim embed_corpus.py
# Set the desired paths/parameters within embed_corpus file
python3 embed_corpus.py 

# Collect data for any benchmark
cd retrievers/[desired_benchmark]

# Set the desired paths/parameters within *_collect file
python3 [desired_benchmark]_collect.py
```

These steps ensure that you have all the embeddings and data required for the benchmarking process in the local environment.

---

### AWS Deployment

For **AWS deployment**, follow these steps to set up and run the pipeline on AWS infrastructure.

#### 1. Manually Configuring the EC2 AWS Instances for Deployment

For **AWS manual deployment**, you will need to configure **EC2 instances** to handle different tasks like embedding, retrieval, and inference.

1. **Create an AWS EC2 Instance**:
   - Log in to your [AWS Console](https://aws.amazon.com/console/).
   - Navigate to the **EC2 Dashboard** and click on **Launch Instance**.
   - Choose an **instance type** (e.g., `r4.8xlarge` for similarity search, `p3.4xlarge` for GPU-based inference).
   - Configure instance details and networking. Ensure that your instances are in a VPC with access to **S3** and other relevant services.
   - Add a **Security Group** that allows SSH access (port 22).
   - Generate or use an existing **SSH Key Pair** to securely access your instances.

2. **Install Dependencies on EC2**:
   - SSH into your EC2 instance once it’s running:
     ```bash
     ssh -i path/to/your-key.pem ec2-user@<your-ec2-public-ip>
     ```
   - Install the required packages:
     ```bash
     git clone https://github.com/your-repository.git
     cd your-repository
     pip install -r requirements.txt
     ```

3. **Upload Passages from the Dataset to S3**:
   - Create an **S3 bucket** in your AWS Console.
   - Upload the passages (e.g., from PubMed or BioASQ) into the S3 bucket in a format suitable for your retrieval system (e.g., JSONL, text files).

4. **Set Up the Vector Database on the EC2 Instance**:
   - For performing similarity searches, set up the vector database (e.g., FAISS Index, HNSW, or Postings lists).
   - **Mount NVMe Volume** (if applicable):
     - AWS EC2 instances can be equipped with NVMe storage, which can be used for storing and accessing the vector database.
     - Attach additional **EBS** volumes and mount them to your EC2 instance.
   - **Store Representations on NVMe**:
     - Upload the vector representations to the NVMe storage via SCP or AWS S3.
     - Update your scripts to read from the NVMe path.

5. **Run the Benchmark on AWS**:
   Once the environment is set up, you can run the benchmark with the code in **benchmarks/local_deployment**. The configuration will be largely the same as local, but you may want to optimize for the cloud environment (e.g., by utilizing multiple instances for parallelized computation).

---

### Using SageMaker for Automated Instance Management

In addition to **manual EC2 setup**, you can use **Amazon SageMaker** to handle the deployment of the RAG pipeline in a fully managed environment. SageMaker allows you to scale instances based on your needs, automate the training and inference pipeline, and optimize for cost and performance.

#### 1. Set Up SageMaker

1. **Create a SageMaker Notebook Instance**:
   - Log in to the **SageMaker Console**.
   - Create a **Notebook Instance** that will automatically scale to the resource requirements of the pipeline.
   - Choose the appropriate **instance type** based on the computational resources required for embedding and retrieval.

2. **Install Dependencies**:
   - On the SageMaker Notebook, install the required packages by running:
     ```bash
     pip install -r requirements.txt
     ```

3. **Upload the Dataset to SageMaker**:
   - Upload the dataset (e.g., BioASQ or PubMed passages) to an **S3 bucket** that will be used by SageMaker for retrieval.

4. **Configure Retrieval**:
   - As with the manual EC2 setup, configure the retrieval model (e.g., BM25, Doc2Vec, SPLADE, or GTR) and the vector database to store representations.
   - SageMaker automatically handles the scaling, so you can adjust the number of instances based on the workload.

#### 2. Scaling with SageMaker

SageMaker allows you to use features like **multi-instance deployment**, which will automatically scale based on the computational load of your benchmarking tasks.

- You can configure **auto-scaling policies** to adjust the number of instances dynamically based on query load or other metrics.
- SageMaker also integrates well with **Elastic Inference** to attach GPU resources for inference tasks without requiring you to manage GPUs manually.

Once the instances are set up, you can run the benchmark using the same steps as local deployment, but SageMaker will handle the scaling and instance management automatically.

---

### 3. Run the Benchmark

In each of the deployment scenarios, once the environment is set up (whether locally, on EC2, or using SageMaker), you can run the benchmark code to measure the performance of your **RAG** pipeline.

- Run the benchmark using the following command in the **benchmarks/local_deployment** directory (or the corresponding directory for AWS/SageMaker):
  ```bash
  python benchmark_pipeline.py
  ```

This will run the entire pipeline and output the performance metrics, including **latency**, **memory usage**, and **throughput**.
