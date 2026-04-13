import requests
import time

def load_model(model_name: str):
    
    print(f"loading model {model_name}")
    
    def model_status():
        status = requests.get("http://llama_cpp:8080/models").json()
        model_status = next(x for x in status["data"] if x["id"] == model_name)
        if not model_status:
            raise Exception("Model is not in list")
        return model_status["status"]["value"]
    
    current_status = model_status()
    if current_status == "loaded":
        print(f"Model {model_name} is already loaded")
        return
    
    if current_status == "unloaded":
        requests.post("http://llama_cpp:8080/models/load", json={
            "model": model_name
        })
    
    while model_status() != "loaded":
        time.sleep(5)

    print(f"Loaded model {model_name}")

def create_embeddings_llama_cpp(text):
    data = {"model": "paraphrase-multilingual", "input": text, "encoding_format": "float"}
    response = requests.post("http://llama_cpp:8080/v1/embeddings", json=data).json()
    return [x["embedding"] for x in response["data"]]

# load_model("paraphrase-multilingual")
# create_embeddings_llama_cpp(["Hello", "yes"])