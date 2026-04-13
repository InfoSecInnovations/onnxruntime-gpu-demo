from test_llama_cpp import load_model, prompt_llama_cpp
from ollama_functionality import load_model_ollama, prompt_ollama
import time

load_model("mistral7b")
load_model_ollama("mistral")

BENCH_ITERATIONS = 5

def bench_llm(func, label):
    total_time = 0
    # run a "mock" prompt to make sure the system is "warmed up"
    func("What does Shabti mean?")
    for i in range(BENCH_ITERATIONS):
        start = time.perf_counter()
        func("What does Shabti mean?")
        end = time.perf_counter()
        total_time += end - start
    print(f"{label} took {total_time / BENCH_ITERATIONS}s on average")

bench_llm(prompt_ollama, "Ollama")
bench_llm(prompt_llama_cpp, "Llama.cpp")