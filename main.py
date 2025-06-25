import threading

from sentence_transformers import SentenceTransformer

from sbert import SentenceBert
from ollama import ollama
from property import Property
from vector_db import VectorDB

vdb = VectorDB()
model = SentenceTransformer('shibing624/text2vec-base-chinese')

vdb.create_collection()


def worker():
    prop = Property()
    prop.generate_property()
    description = ollama.generate_description(prop.to_prompt())
    prop.description = description[:1024]

    p_dict = prop.to_dict()
    p_dict["desc_vector"] = SentenceBert.text2vector(model, prop.description)
    vdb.upsert([p_dict])


def prepare_data(num=100):
    threads = []
    for i in range(num):
        thread = threading.Thread(target=worker)
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()


if __name__ == '__main__':
    # prop = Property()
    # prop.generate_property()
    # description = ollama.generate_description(prop.to_prompt())
    # print(description)
    # exit(0)
    while True:
        prepare_data(100)
        print("prepare data 100")
