import threading

from sentence_transformers import SentenceTransformer

from sbert import SentenceBert
from ollama import ollama
from property import Property
from vector_db import VectorDB

# 训练教程？
# https://huggingface.co/blog/train-sentence-transformers#trainer

# 语义搜索模型排行榜
# https://huggingface.co/spaces/mteb/leaderboard

# 语义模型趋势榜
# https://huggingface.co/models?pipeline_tag=sentence-similarity&language=zh&sort=trending

vdb = VectorDB()
# model = SentenceTransformer('shibing624/text2vec-base-chinese')
model1 = SentenceTransformer('Alibaba-NLP/gte-multilingual-base',trust_remote_code= True)
model2 = SentenceTransformer('HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v2')

# vdb.create_collection()


def worker():
    prop = Property()
    prop.generate_property()
    # description = ollama.generate_description(prop.to_prompt())
    # prop.description = description[:1024]

    p_dict = prop.to_dict()
    p_dict["desc_vector1"] = SentenceBert.text2vector(model1, prop.description)
    p_dict["desc_vector2"] = SentenceBert.text2vector(model2, prop.description)

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
    # print(prop.to_prompt())
    # print(SentenceBert.text2vector(model2, prop.to_prompt()))
    # description = ollama.generate_description(prop.to_prompt())
    # print(description)
    # exit(0)
    while True:
        prepare_data(100)
        print("prepare data 100")
