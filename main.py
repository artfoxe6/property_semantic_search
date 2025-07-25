import csv
import os
import sys
import time
from pathlib import Path

import train
from sbert import SentenceBert
from property import Property
from sqlite_db import SqliteDB
from train import train_model
from vector_db import VectorDB

# 训练教程？
# https://huggingface.co/blog/train-sentence-transformers#trainer
# https://sbert.net/docs/sentence_transformer/training_overview.html

# 语义搜索模型排行榜
# https://huggingface.co/spaces/mteb/leaderboard

# 语义模型趋势榜
# https://huggingface.co/models?pipeline_tag=sentence-similarity&language=zh&sort=trending

# https://huggingface.co/Alibaba-NLP/gte-multilingual-base
sdb = None
vdb = None

def worker(b: SentenceBert, v: VectorDB):
    p = Property()
    p.random_value()
    p_dict = p.to_dict()
    p_dict["desc_vector"] = b.text2vector(p.description)

    v.upsert([p_dict])


def gen_milvus_data(num=1000):
    v_db = VectorDB(True)
    bert = SentenceBert()
    id = 0
    count = 0
    while True:
        p = Property(id)
        p.random_value()
        p_dict = p.to_dict()
        p_dict["desc_vector"] = bert.text2vector(p.description)
        v_db.upsert([p_dict])
        count += 1
        id += 1
        if count % (num / 10) == 0:
            print(f"{count}/{num}")
        if count >= num:
            break


def gen_property_data(train_num=10000, milvus_num=10000):
    print(f"gen property data {train_num}")
    s_db = SqliteDB("property.db")
    count = 0
    while True:
        prop = Property()
        prop.random_value()
        s_db.add_property(prop)
        count +=1
        if count % (train_num / 10) == 0:
            print(f"{count}/{train_num}")
        if count >= train_num:
            break

    print(f"gen property data {milvus_num}")
    s_db = SqliteDB("property_milvus.db")
    count = 0
    while True:
        prop = Property()
        prop.random_value()
        s_db.add_property(prop)
        count +=1
        if count >= milvus_num:
            break
        if count % (milvus_num / 10) == 0:
            print(f"{count}/{milvus_num}")


def sync_to_milvus(num=10000):
    v_db = VectorDB(True)
    # s_db = SqliteDB("property_milvus.db")
    s_db = SqliteDB("property.db")
    page_size = 1000
    last_id = 0
    count = 0
    b = SentenceBert()
    while True:
        props = s_db.list(last_id, page_size)
        if not props:
            break
        for prop in props:
            p_dict = prop.to_dict()
            p_dict["desc_vector"] = b.text2vector(prop.description)

            v_db.upsert([p_dict])
            last_id = prop.id
            count += 1
            if count % (num / 10) == 0:
                print(f"{count}/{num}")
        if count >= num:
            break


def gen_training_data(tran_count=10000, dev_count=1000):
    print("gen training data")
    s_db = SqliteDB()
    page_size = 1000
    last_id = 0
    header = ["query", "positive", "negative"]

    fp_dev = open("train_data_dev.csv", "w", newline="", encoding="utf-8")
    writer_dev = csv.writer(fp_dev)
    writer_dev.writerow(header)

    fp = open("train_data.csv", "w", newline="", encoding="utf-8")
    writer = csv.writer(fp)
    writer.writerow(header)

    count_dev = 0
    count = 0
    while True:
        props = s_db.list(last_id, page_size)
        if not props:
            break
        for prop in props:
            last_id = prop.id
            queries = prop.property_to_query_texts_v2()
            for query in queries:
                if count < tran_count:
                    negative = prop.gen_negative_property(query[0])
                    writer.writerow([query[1], prop.description, negative])
                    count += 1
                    if count % (tran_count / 10) == 0:
                        print(f"{count}/{tran_count}")
                elif count_dev < dev_count:
                    negative = prop.gen_negative_property(query[0])
                    writer_dev.writerow([query[1], prop.description, negative])
                    count_dev += 1
                    if count_dev % (dev_count / 10) == 0:
                        print(f"dev {count_dev}/{dev_count}")
                else:
                    break

        if count_dev >= dev_count:
            break
    fp.close()
    fp_dev.close()


# PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python main.py all
if __name__ == '__main__':
    train_num = 1000000
    step = ""
    if len(sys.argv) == 2:
        step = sys.argv[1]

    if step == "gen_property_data":
        gen_property_data(train_num, train_num)
    elif step == "gen_training_data":
        gen_training_data(train_num, int(train_num/10))
    elif step == "train":
        train_model(model_name='./paraphrase-multilingual-MiniLM-L12-v2')
    elif step == "gen_milvus_data":
        gen_milvus_data(int(train_num/10))
    elif step == "sync_to_milvus":
        sync_to_milvus(int(train_num/10))
    elif step == "all":
        try:
            os.remove("property.db")
            os.remove("property_milvus.db")
            old_folder = Path("train_model")
            new_folder = Path(f"train_model{time.time()}")
            old_folder.rename(new_folder)
        except Exception as e:
            print(e)
        finally:
            pass
        gen_property_data(train_num, train_num)
        gen_training_data(train_num, int(train_num/10))
        # train_model(model_name='./gte-multilingual-base')
        train_model(model_name='./paraphrase-multilingual-MiniLM-L12-v2')
        # gen_milvus_data(10000)
    else:
        train_model(model_name='./paraphrase-multilingual-MiniLM-L12-v2')
        print("Usage: python main.py xxxx")