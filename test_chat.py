from test_llama_cpp import load_model, prompt_llama_cpp
from ollama_functionality import load_model_ollama, prompt_ollama

load_model("mistral7b")
load_model_ollama("mistral")

print("----llama.cpp----\n\n")

print(prompt_llama_cpp("What does Shabti mean?"))

print("\n\n----Ollama----\n\n")

print(prompt_ollama("What does Shabti mean?"))