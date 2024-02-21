import tinyvectorstore.tinyvstore as tinyvstore


def test_tiny_vector_usage():
    max_tokens_count = 768
    tinyvc = tinyvstore.TinyVectorStore(name="short_term_logs",
                                        max_tokens=max_tokens_count,
                                        persist_data=True,
                                        data_storage_path=None)

    print("\n==BUILD CORPUS==" * 1, "\n")
    corpus = [
        "I love eat mango, mango is my favorite fruit. so, buy lots of mangon today because we need eat lots of healthy food.",
        "mango is my favorite fruit",
        "mango, apple, oranges are fruits",
        "mango, mango, mango, mango fruit",
        "fruits are good for health",
    ]
    corpus = tinyvc.chunk_sentences_to_max_tokens(
        sentences=corpus, max_tokens=max_tokens_count)
    corpus_vectors = tinyvc.encode(corpus)
    tinyvc.insert_vectors(corpus_vectors)
    print("\n==Similarity Search==" * 1, "\n")
    query = "mango fruit"
    result = tinyvc.similarity_search(query)
    print(result[0])
    print("\n==Persistance to Disk==" * 1, "\n")
    tinyvc.save_to_disk()


def test_tiny_vector_load_persist_data():
    max_tokens_count = 768
    tinyvc = tinyvstore.TinyVectorStore.get_or_create_vectorstore(name="short_term_logs",
                                                                  storage_path=None,
                                                                  max_tokens=max_tokens_count)
    query = "mango fruit"
    result = tinyvc.similarity_search(query)
    print(result)


if __name__ == "__main__":
    test_tiny_vector_usage()
    test_tiny_vector_load_persist_data()
