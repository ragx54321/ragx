import time
import pynvml
import torch
import tensorrt_llm as trtllm
from transformers import LlamaTokenizer

class Llama65BInference:
    def __init__(self):
        self.model_size = "65b"
        engine_path = "llama_models/llama-65b.engine" 
        self.tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-65b-hf")
        self.model = trtllm.TRTLLM(engine_path)
        
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        
        torch.cuda.set_device(0)
    
    def get_energy(self):
        return pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
    
    def infer(self, query):
        inputs = self.tokenizer(query, return_tensors="pt").input_ids.cuda()
        input_length = inputs.shape[1]
        
        energy_before = self.get_energy()
        start_time = time.time()
        
        with torch.no_grad():
            output = self.model.generate(inputs, max_length=128)
        
        end_time = time.time()
        energy_after = self.get_energy()
        
        latency = end_time - start_time
        energy_consumed = (energy_after - energy_before) / 1000.0
        output_tokens = output.shape[1]
        throughput = output_tokens / latency
        
        output_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        return {
            "query": query,
            "model_size": self.model_size,
            "input_length": input_length,
            "latency": latency,
            "energy": energy_consumed,
            "output_tokens": output_tokens,
            "throughput": throughput,
            "output_text": output_text,
        }
    
    def save_engine(self):
        engine_output_path = "llama_models/llama-65b-optimized.engine"
        self.model.save(engine_output_path)
        print(f"Saved optimized engine to {engine_output_path}")
    
    def __del__(self):
        pynvml.nvmlShutdown()

if __name__ == "__main__":
    print("Running inference for LLaMA-65B on single GPU")
    llama = Llama65BInference()
    test_query = "What is the role of mitochondria in cellular respiration?"
    result = llama.infer(test_query)
    print(result)
    llama.save_engine()
