# LLaMA 2 Inference with TensorRT-LLM

The implementation includes profiling for latency, energy consumption, throughput, and optimizations such as Flash Attention.

## Prerequisites
Before using this repository, you **must request access to Meta's LLaMA 2 models**:
[Request Access to LLaMA 2](https://ai.meta.com/resources/models-and-libraries/llama-downloads/)

Once granted access, download the model weights and place them in the appropriate directory (`llama_models/`).

## Installation
Ensure your environment meets the following requirements:
- Python 3.8+
- CUDA 11.8+
- NVIDIA TensorRT-LLM
- PyTorch with GPU support
- Transformers
- pynvml

### Install dependencies
Create a virtual environment and install the required dependencies:
```bash
python -m venv llama_env
source llama_env/bin/activate  # On Windows use: llama_env\Scripts\activate
pip install -r requirements.txt
```

### `requirements.txt`
```
torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
transformers
nvidia-pyindex
tensorrt
pynvml
```

## LLaMA 2 Model Variants
| Model   | Parameters | Transformer Layers | GPUs Required |
|---------|-----------|--------------------|---------------|
| LLaMA 2 13B  | 13B  | 40 | 1 GPU |
| LLaMA 2 33B  | 33B  | 60 | 2 GPUs |
| LLaMA 2 65B  | 65B  | 80 | 4 GPUs |

## Flash Attention Optimization
Flash Attention is used to improve memory efficiency and speed up transformer-based inference. The implementation enables it for LLaMA 33B and larger models.

## Running Inference
### Example for LLaMA 13B
```bash
python llama_13b_inference.py --query "What is the role of mitochondria in cellular respiration?"
```
### Example for LLaMA 33B with Flash Attention
```bash
python llama_33b_inference.py --query "Explain quantum entanglement."
```

## Deployment on AWS
To deploy on AWS, you will need a GPU-optimized instance. Follow these steps:

### Renting a DGX Machine
1. Sign in to [AWS EC2 Console](https://aws.amazon.com/ec2/)
2. Select **Launch Instance**
3. Choose a GPU-enabled instance such as:
   - `p4d.24xlarge` (8x A100 GPUs, 320 GB GPU RAM)
   - `g5.48xlarge` (4x A10G GPUs, 96 GB GPU RAM)
4. Use the **Deep Learning AMI** (Amazon Machine Image) for pre-installed CUDA and PyTorch.
5. Configure storage and networking.
6. Launch the instance and SSH into it:
```bash
ssh -i my-key.pem ubuntu@your-ec2-instance-ip
```

### Running LLaMA on AWS
Once connected:
```bash
git clone https://github.com/your-repo/llama-inference.git
cd llama-inference
pip install -r requirements.txt
python llama_33b_inference.py --query "How does photosynthesis work?"
```

## Parallelization and GPU Utilization
- **Single GPU (13B Model)**: Runs on one A100 or equivalent.
- **Multi-GPU (33B, 65B Models)**: Utilizes `torch.nn.DataParallel` to distribute computation across GPUs.
- **TensorRT Optimization**: Reduces latency via kernel fusion and efficient tensor computations.
