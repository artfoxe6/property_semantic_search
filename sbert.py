import sys

from sentence_transformers import SentenceTransformer


# model = SentenceTransformer('shibing624/text2vec-base-chinese')
# model1 = SentenceTransformer('Alibaba-NLP/gte-multilingual-base', trust_remote_code=True)
# model2 = SentenceTransformer('HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v2')


class SentenceBert:
    def __init__(self):
        self.model1 = SentenceTransformer('Alibaba-NLP/gte-multilingual-base', trust_remote_code=True)
        self.model2 = SentenceTransformer('HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v2')

    def text2vector(self, model_type: int, text):
        model = self.model1 if model_type == 1 else self.model2
        embedding = model.encode(text, normalize_embeddings=True)
        return embedding


if __name__ == "__main__":
    queryStr = sys.argv[1]
    sbert = SentenceBert()

    print("model1:")
    print(",".join(map(str, sbert.text2vector(1, queryStr))))
    print("model2:")
    print(",".join(map(str, sbert.text2vector(2, queryStr))))
