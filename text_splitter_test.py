from semantic_text_splitter import TextSplitter
from tokenizers import Tokenizer
from pprint import pprint

tokenizer = Tokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
splitter = TextSplitter.from_huggingface_tokenizer(tokenizer, tokenizer.truncation['max_length'], overlap=50)

with open("bench_text.txt") as f:
    text = f.read()
split = splitter.chunks(text)
print(len(split))