import threading

from sbert import SentenceBert
from property import Property
from vector_db import VectorDB

# 训练教程？
# https://huggingface.co/blog/train-sentence-transformers#trainer

# 语义搜索模型排行榜
# https://huggingface.co/spaces/mteb/leaderboard

# 语义模型趋势榜
# https://huggingface.co/models?pipeline_tag=sentence-similarity&language=zh&sort=trending

vdb = VectorDB()
# vdb.create_collection()


def worker(bert: SentenceBert):
    p = Property()
    p.generate_property()
    p_dict = p.to_dict()
    p_dict["desc_vector1"] = bert.text2vector(1, p.description)
    p_dict["desc_vector2"] = bert.text2vector(2, p.description)

    vdb.upsert([p_dict])


def prepare_data(bert: SentenceBert, num=100):
    threads = []
    for i in range(num):
        thread = threading.Thread(target=worker, args=([bert]))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()


if __name__ == '__main__':
    prop = Property()
    prop.generate_property()
    print(prop.combine_description())
    exit(0)
    # print(prop.to_prompt())
    # print(SentenceBert.text2vector(model2, prop.to_prompt()))
    # description = ollama.generate_description(prop.to_prompt())
    # print(description)
    # exit(0)
    b = SentenceBert()
    while True:
        prepare_data(b, 100)
        print("prepare data 100")
