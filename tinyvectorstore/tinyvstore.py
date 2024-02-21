import logging
import os
import shutil
import joblib
from joblib import Memory
import numpy as np

logging.basicConfig(
    level=logging.WARN,
    filename="tinyvectorstore.log",
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class TinyVectorStore:

    _DATA_STORAGE_PATH = "./tvdb"
    _CACHE_PATH = "./tvdb/cache"
    _DATA_STORAGE_INDEX = "tvi"
    _DATA_STORAGE_DATA = "tvd"

    def __init__(
        self,
        name: str = None,
        max_tokens: int = 768,
        cache_data: bool = True,
        persist_data: bool = False,
        data_storage_path: str = None,
        cache_path: str = None,
    ):
        self._name = name
        self._vector_data = {}  # store vectors dataset
        self._vector_index = {}  # indexing structure for retrieval
        self._vocabulary = set()  # unique list of vocabulary words
        self._word_to_index = {}  # mapping word to index id
        self._max_tokens = max_tokens  # max size of the arrray
        self._persist_data = persist_data
        self._data_storage_path = data_storage_path
        if self._persist_data:
            self._init_persistance(data_storage_path, cache_path)

        print("TinyVectorStore._max_tokens:", self._max_tokens)
        if cache_data:
            if cache_path is None:
                self._cache_path = self._CACHE_PATH
            os.makedirs(self._cache_path, exist_ok=True)
            # use memory mapping to speeds up cache looking when reloading large numpy arrays
            self._memory = Memory(self._CACHE_PATH, mmap_mode='r')
            self._cosine_similarity = self._memory.cache(
                self._cosine_similarity)
            self.encode = self._memory.cache(self.encode)
            self.similarity_search = self._memory.cache(self.similarity_search)

    def _init_persistance(self, data_storage_path: str = None, cache_path: str = None):
        if data_storage_path is None:
            self._data_storage_path = str(self._DATA_STORAGE_PATH)
        if cache_path is None:
            self._cache_path = self._CACHE_PATH
        try:
            os.makedirs(self._data_storage_path, exist_ok=True)
            os.makedirs(self._cache_path, exist_ok=True)
            print(
                f"TinyVectorStore data directory created at {self._data_storage_path}"
            )
            print(
                f"TinyVectorStore cache directory created at {self._cache_path}")
        except OSError as error:
            print("TinyVectorStore data directory can not be created")

    def add_vector(self, vector_id, vector):
        logger.debug(f"add_vector id: {vector_id} | vector: {vector}")
        self._vector_data[vector_id] = vector
        self._update_index(vector_id, vector)

    def get_vector(self, vector_id):
        vector = self._vector_data.get(vector_id)
        logger.debug(f"get_vector id: {vector_id} \n vector: {vector}")
        return vector

    def _cosine_similarity(self, vector1, vector2):
        return (
            np.dot(vector1, vector2)
            / (np.linalg.norm(vector1))
            * np.linalg.norm(vector2)
        )

    def _update_index(self, vector_id, vector):
        logger.debug(f"_update_index id: {vector_id} \n vector: {vector}")
        for existing_id, existing_vector in self._vector_data.items():
            similarity = self._cosine_similarity(vector, existing_vector)
            if existing_id not in self._vector_index:
                self._vector_index[existing_id] = {}
                logger.warning(
                    f"_update_index id: index not found [{existing_id}]")
            logger.info(
                f"_update_index id: apply update on existing index: similarity[{similarity}]"
            )
            self._vector_index[existing_id][vector_id] = similarity

    def find_similar_vectors(self, query_vector, num_results=5):
        results = []
        for vector_id, vector in self._vector_data.items():
            similarity = self._cosine_similarity(query_vector, vector)
            results.append((vector_id, similarity))
        # Sort by similarity in descending order
        results.sort(key=lambda x: x[1], reverse=True)
        # Return the top N results
        logger.debug(
            f"find_similar_vectors : query vector {query_vector} \n results: {results[:num_results]}")
        return results[:num_results]

    def text_to_vector(self, sentence: str):
        self._update_vocabulary_and_index([sentence])
        tokens = sentence.lower().split()
        if self._max_tokens < len(self._word_to_index):
            raise ValueError(
                f"max tokens can not be smaller than words index size: {len(self._word_to_index)}")
        # static vector size
        vector = np.zeros(self._max_tokens)
        for token in tokens:
            vector[self._word_to_index[token]] += 1
        return vector

    @staticmethod
    def sentence_word_splitter(num_of_words: int, sentence: str) -> list:
        pieces = sentence.split()
        return [" ".join(pieces[i:i+num_of_words]) for i in range(0, len(pieces), num_of_words)]

    @staticmethod
    def chunk_text_to_fixed_length(text: str, length: int):
        text = text.strip()
        result = [text[0+i:length+i] for i in range(0, len(text), length)]
        return result

    def chunk_sentences_to_fixed_length(self, sentences: list, max_length: int = 768):
        fixed_size_sentences = []
        for sentence in sentences:
            chunks = TinyVectorStore.chunk_text_to_fixed_length(
                text=sentence, length=max_length)
            fixed_size_sentences = fixed_size_sentences+chunks
        return fixed_size_sentences

    def chunk_sentences_to_max_tokens(self, sentences: list, max_tokens: int = 768):
        fixed_size_sentences = []
        for sentence in sentences:
            tokens = sentence.lower().split()
            if len(tokens) > (max_tokens):
                chunks = TinyVectorStore.sentence_word_splitter(
                    num_of_words=max_tokens, sentence=sentence)
                fixed_size_sentences = fixed_size_sentences+chunks
            else:
                fixed_size_sentences.append(sentence)
        return fixed_size_sentences

    def sentences_to_vector(self, sentences: list) -> list:
        logger.debug(f"_text_to_vector:\n")
        sentence_vectors = {}
        for sentence in sentences:
            vector = self.text_to_vector(sentence)
            sentence_vectors[sentence] = vector
        return sentence_vectors

    def _update_vocabulary_and_index(self, sentences: list):
        logger.debug(f"_update_vocabolary_and_index")
        for sentence in sentences:
            tokens = sentence.lower().split()
            self._vocabulary.update(tokens)
        # assign unique id to words in the vocabulary
        self._word_to_index = {word: i for i,
                               word in enumerate(self._vocabulary)}
        logger.debug(
            f"vocabulary: {self._vocabulary} \n word_to_index: {self._word_to_index}"
        )

    def encode(self, sentences: list):
        logger.debug(f"encode_text")
        self._update_vocabulary_and_index(sentences)
        sentence_vectors = self.sentences_to_vector(sentences)
        logger.debug(
            f"<sentence_vectors>\n {sentence_vectors} \n</sentence_vectors>")
        return sentence_vectors

    def save_text(self, key: str, content: str):
        logger.debug(f"save_text:\n")
        self.add_vector(vector_id=key, vector=content)

    def insert_vectors(self, sentence_vectors: list):
        logger.debug(f"save_vectors:\n")
        for key in sentence_vectors.keys():
            self.add_vector(vector_id=key, vector=sentence_vectors[key])

    def dump(self):
        print(f"-------------------" * 5)
        print(f"<vectorstore>")
        print(f"<vocabulary>{self._vocabulary}</vocabulary>")
        print(f"<word_to_index>{self._word_to_index}</word_to_index>")
        print(f"<vector_index>{self._vector_index}</vector_index>")
        print(f"<vector_data>{self._vector_data}</vector_data>")
        print(f"</vectorstore>\n")
        print(f"-------------------" * 5)

    def similarity_search(self, query: str, num_results: int = 3):
        logger.debug(f"similarity_search query:{query}\n")
        query_vector = self.text_to_vector(query)
        similar_sentences = self.find_similar_vectors(
            query_vector, num_results=num_results
        )
        result = []
        for sentence, similarity in similar_sentences:
            result.append([sentence, f"{similarity:.4f}"])
        logger.debug(f"similar sentences found: {result}")
        return result

    def save_to_disk(self):
        self._init_persistance(self._data_storage_path, self._cache_path)
        save_path = datafile = self._data_storage_path+"/"+self._name
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        datafile = save_path+"/"+self._name
        joblib.dump(self._vector_data, datafile +
                    ".data.gz", compress=('gzip', 3))
        joblib.dump(self._vector_index, datafile +
                    ".index.gz", compress=('gzip', 3))
        joblib.dump(self._vocabulary, datafile +
                    ".vocabulary.gz", compress=('gzip', 3))
        joblib.dump(self._word_to_index, datafile +
                    ".word_to_index.gz", compress=('gzip', 3))
        with open(datafile+".hash", "w", newline="") as file:
            file.write(str(joblib.hash(self)))
        print(f"TinyVectorstore data files {datafile} saved to disk!")

    @classmethod
    def load_from_disk(cls, name: str = "default", data_storage_path: str = None, max_tokens: int = 768):
        if data_storage_path is None:
            datafile = str(cls._DATA_STORAGE_PATH)+"/"+name
        else:
            datafile = str(data_storage_path)+"/"+name+"/"+name
        print("loading directory ", datafile)
        if os.path.exists(datafile+".data.gz"):
            instance = cls(max_tokens=max_tokens)
            instance._vector_data = joblib.load(datafile+".data.gz")
            instance._vector_index = joblib.load(datafile+".index.gz")
            instance._vocabulary = joblib.load(datafile+".vocabulary.gz")
            instance._word_to_index = joblib.load(datafile+".word_to_index.gz")
            print("TinyVectorstore data file loaded!")
            return instance
        else:
            print("Missing TinyVectorstore data file!")
            return cls()

    @classmethod
    def delete_from_disk(cls, data_storage_path: str = None, confirm_deletion: str = "N"):
        if confirm_deletion == "Y":
            print("delete persistance directory from disk...")
            if data_storage_path is None:
                raise ValueError("Missing TinyVectorstore data file!")
            shutil.rmtree(data_storage_path)
            print(f"deleted from disk: {data_storage_path}")
        else:
            raise ValueError("please confirm deletion by setting flag to Y")

    @staticmethod
    def get_or_create_vectorstore(name: str = "default", storage_path: str = None, max_tokens: int = 768):
        """get or create tiny vectorstore"""
        if storage_path is None:
            storage_path = "./tvdb"
        local_storage_path = storage_path+"/" + name
        if os.path.exists(local_storage_path):
            tinyvc = TinyVectorStore.load_from_disk(
                name=name, data_storage_path=storage_path, max_tokens=max_tokens)
        else:
            tinyvc = TinyVectorStore(name=name, max_tokens=max_tokens,
                                     persist_data=True, data_storage_path=local_storage_path)
        return tinyvc


# def test_tiny_vector_usage():
#     max_tokens_count = 768
#     tinyvc = TinyVectorStore(name="short_term_logs",
#                              max_tokens=max_tokens_count,
#                              persist_data=True,
#                              data_storage_path=None)

#     print("\n=======CORPUS======" * 1, "\n")
#     corpus = [
#         "I love eat mango, mango is my favorite fruit. so, buy lots of mangon today because we need eat lots of healthy food.",
#         "mango is my favorite fruit",
#         "mango, apple, oranges are fruits",
#         "mango, mango, mango, mango fruit",
#         "fruits are good for health",
#     ]
#     corpus = tinyvc.chunk_sentences_to_max_tokens(
#         sentences=corpus, max_tokens=max_tokens_count)
#     corpus_vectors = tinyvc.encode(corpus)
#     tinyvc.insert_vectors(corpus_vectors)
#     print("\n=======Similarity Search======" * 1, "\n")
#     query = "mango fruit"
#     result = tinyvc.similarity_search(query)
#     print(result[0])
#     print("\n======SAVE======" * 1, "\n")
#     tinyvc.save_to_disk()


# def test_tiny_vector_load_persist_data():
#     max_tokens_count = 768
#     tinyvc = TinyVectorStore.get_or_create_vectorstore(name="short_term_logs",
#                                                        storage_path=None,
#                                                        max_tokens=max_tokens_count)
#     query = "mango fruit"
#     result = tinyvc.similarity_search(query)
#     print(result)


# if __name__ == "__main__":
#     test_tiny_vector_usage()
#     test_tiny_vector_load_persist_data()
