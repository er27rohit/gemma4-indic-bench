import time
import json
import pathlib
from vllm import LLM, SamplingParams

def run_benchmark(model_id="google/gemma-4-e2b-it"):
    # Initialize vLLM engine
    llm = LLM(model=model_id, max_model_len=4096, gpu_memory_utilization=0.85)
    sampling_params = SamplingParams(max_tokens=256, temperature=0.3)

    # Indic-specific prompts
    prompts = [
        "Explain KV Caching in 100 words in Hindi.",
        "SystemVerilog Assertion for a simple FIFO: Write the code.",
        "How to optimize C++ memory for LLM inference?",
    ]

    results = []
    print(f"--- Starting Benchmark for {model_id} ---")

    for p in prompts:
        start_time = time.perf_counter()
        outputs = llm.generate([p], sampling_params)
        end_time = time.perf_counter()
        
        latency = end_time - start_time
        generated_text = outputs[0].outputs[0].text
        
        results.append({
            "prompt": p,
            "latency_sec": latency,
            "char_count": len(generated_text)
        })
        print(f"Prompt: {p[:30]}... | Latency: {latency:.2f}s")

    # Save results
    output_path = pathlib.Path("results/latency_results.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    run_benchmark()
