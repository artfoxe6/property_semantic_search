import sys

from sentence_transformers import SentenceTransformer, util


# model = SentenceTransformer('shibing624/text2vec-base-chinese')
# model1 = SentenceTransformer('Alibaba-NLP/gte-multilingual-base', trust_remote_code=True)
# model2 = SentenceTransformer('HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v2')


class SentenceBert:
    def __init__(self):
        # self.model = SentenceTransformer('./gte-multilingual-base',trust_remote_code=True)
        self.model = SentenceTransformer('./train_model',trust_remote_code=True)
        # self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def text2vector(self, text):
        return self.model.encode(text, normalize_embeddings=True)

    # 计算两个query的相似度
    def similarity(self, query1, query2):
        embedding1 = self.model.encode(query1, convert_to_tensor=True)
        embedding2 = self.model.encode(query2, convert_to_tensor=True)

        similarity = util.pytorch_cos_sim(embedding1, embedding2)
        print(similarity.item())


if __name__ == "__main__":
    if len(sys.argv) == 2:
        queryStr = sys.argv[1]
        sbert = SentenceBert()

        print("["+",".join(map(str, sbert.text2vector(queryStr)))+"]")
        sys.exit(1)
    elif len(sys.argv) == 3:
        sbert = SentenceBert()
        sbert.similarity(sys.argv[1], sys.argv[2])
