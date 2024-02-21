# tiny-vstore
implementation of vector store build from scratch with minimal dependencies for embedding data and perform similarity search.

### Use Case
Simple solution for embedding vector storage with persistance support and least dependencies. 


### Run Locally
require poetry install
```
git clone https://github.com/PavAI-Research/tiny-vstore.git

poetry install 

```

### Usage
```python

## tokens size
max_tokens_count = 768

tinyvc = tinyvstore.TinyVectorStore(name="short_term_logs",
                                    max_tokens=max_tokens_count,
                                    persist_data=True,
                                    data_storage_path=None)

## BUILD CORPUS==
corpus = [
    "I love eat mango, mango is my favorite fruit. so, buy lots of mangon today because we need eat lots of healthy food.",
    "mango is my favorite fruit",
    "mango, apple, oranges are fruits",
    "mango, mango, mango, mango fruit",
    "fruits are good for health",
]
## Preprocess
corpus = tinyvc.chunk_sentences_to_max_tokens(
    sentences=corpus, max_tokens=max_tokens_count)

## Encode
corpus_vectors = tinyvc.encode(corpus)

## Save
tinyvc.insert_vectors(corpus_vectors)
## ==Similarity Search==
query = "mango fruit"
result = tinyvc.similarity_search(query)
print(result[0])

##=Persistance to disk==
tinyvc.save_to_disk()

```

### Poetry install 
see link here: https://python-poetry.org/docs/#installation

or export requirements

poetry export --without-hashes --format=requirements.txt > requirements.txt

### TODOs

> enhance Dence retrieval support
> add Sparse retrieval support 
> add Multi-vector retrieval support 

