import torch
import onnxruntime
from sentence_transformers import SentenceTransformer
from test_llama_cpp import load_model, create_embeddings_llama_cpp
from ollama_functionality import create_embeddings_ollama, load_model_ollama
from fastembed import TextEmbedding
import os
import time
from langchain_text_splitters import SentenceTransformersTokenTextSplitter

print("torch cuda is available: " + "Yes" if torch.cuda.is_available() else "No")

print("torch cuda version " + torch.version.cuda)
print("torch cudnn version: " + str(torch.backends.cudnn.version()))

print("preload DLLs")
onnxruntime.preload_dlls()
print(onnxruntime.get_available_providers())

stransform = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

model = TextEmbedding(
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    cache_dir=os.getenv("SHABTI_MODELS_DIR"),
    providers=["CUDAExecutionProvider"]
)

load_model_ollama("paraphrase-multilingual")

# load model in llama.cpp
load_model("paraphrase-multilingual")

splitter = SentenceTransformersTokenTextSplitter(
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)


with open("bench_text.txt") as f:
    text = f.read()
split = splitter.split_text(text)
print(len(split))

BENCH_ITERATIONS = 5

def bench_embedding_method(func, label):
    total_time = 0
    for i in range(BENCH_ITERATIONS):
        start = time.perf_counter()
        embeddings = func(split)
        end = time.perf_counter()
        total_time += end - start
    print(f"{label} took {total_time / BENCH_ITERATIONS}s on average")

print("start embeddings bench")

bench_embedding_method(lambda x: list(model.embed(x)), "FastEmbed")
bench_embedding_method(stransform.encode, "Sentence Transformers")
bench_embedding_method(create_embeddings_ollama, "Ollama")
bench_embedding_method(create_embeddings_llama_cpp, "Llama.cpp")

print("end embedding bench")
