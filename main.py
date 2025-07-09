import csv
import sys
import threading

from sympy.strategies.core import switch

from sbert import SentenceBert
from property import Property
from sqlite_db import SqliteDB
from vector_db import VectorDB

# 训练教程？
# https://huggingface.co/blog/train-sentence-transformers#trainer

# 语义搜索模型排行榜
# https://huggingface.co/spaces/mteb/leaderboard

# 语义模型趋势榜
# https://huggingface.co/models?pipeline_tag=sentence-similarity&language=zh&sort=trending
sdb = SqliteDB()
vdb = None


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


def random_property_to_db(num=1000):
    while num > 0:
        prop = Property()
        prop.generate_property()
        sdb.add_property(prop)
        num -= 1

def sync_to_milvus():
    page_size = 1000
    last_id = 0
    b = SentenceBert()
    while True:
        props = sdb.list(last_id, page_size)
        if not props:
            break
        for prop in props:
            p_dict = prop.to_dict()
            p_dict["desc_vector1"] = b.text2vector(1, prop.description)
            p_dict["desc_vector2"] = b.text2vector(2, prop.description)

            vdb.upsert([p_dict])
            last_id = prop.id

def gen_training_data():
    page_size = 1000
    last_id = 0
    header = ["query", "positive", "negative"]
    fp = open("trandata.csv", "w", newline="", encoding="utf-8")
    writer = csv.writer(fp)
    writer.writerow(header)
    while True:
        props = sdb.list(last_id, page_size)
        if not props:
            break
        for prop in props:
            queries = prop.property_to_query_texts()
            for query in queries:
                negative = prop.gen_negative_property(query[0])
                print(negative)
                exit(0)
                writer.writerow([query[1], prop.description, negative])
                last_id = prop.id
    fp.close()


if __name__ == '__main__':
    step = sys.argv[1]

    if step == "prepare":
        random_property_to_db(500000)
    elif step == "sync":
        vdb = VectorDB()
        sync_to_milvus()
    elif step == "gen_tran_data":
        gen_training_data()
    elif step == "train":
        pass
    else:
        p = Property()
        p.generate_property()
        print(p.to_prompt())
        print("Usage: python main.py prepare | train | test")
    # prop = Property()
    # prop.generate_property()
    # print(prop.combine_description())
    # exit(0)
    # print(prop.to_prompt())
    # print(SentenceBert.text2vector(model2, prop.to_prompt()))
    # description = ollama.generate_description(prop.to_prompt())
    # print(description)
    # exit(0)
    # b = SentenceBert()
    # while True:
    #     prepare_data(b, 100)
    #     print("prepare data 100")
