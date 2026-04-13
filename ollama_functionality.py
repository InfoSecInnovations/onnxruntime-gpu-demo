import requests
import json

def create_embeddings_ollama(text):
    data = {"model": "paraphrase-multilingual", "input": text, "stream": False}
    response = requests.post(
        f"http://ollama:11434/api/embed", data=json.dumps(data)
    ).json()
    return response["embeddings"]

def do_loading(model_name):
    def is_loaded():
        models = requests.get(f"http://ollama:11434/api/tags")
        model_list = json.loads(models.text)["models"]
        return next(
            filter(lambda x: x["name"].split(":")[0] == model_name, model_list),
            None,
        )

    while not is_loaded():
        print(f"{model_name} model not found. Please wait while it loads.")
        request = requests.post(
            f"http://ollama:11434/api/pull",
            data=json.dumps({"name": model_name}),
            stream=True,
        )
        current = 0
        for item in request.iter_lines():
            if item:
                value = json.loads(item)
                # TODO: display statuses
                if "total" in value:
                    if "completed" in value:
                        current = value["completed"]
                    yield current

def load_model_ollama(model_name):
    for _ in do_loading(model_name):
        pass
    
def prompt_ollama(input):
    data = {"model": "mistral", "prompt": input, "stream": False}
    response = requests.post(
        f"http://ollama:11434/api/generate", data=json.dumps(data)
    ).json()
    return response["response"]