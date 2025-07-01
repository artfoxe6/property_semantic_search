import sys

from sentence_transformers import SentenceTransformer, util


# model = SentenceTransformer('shibing624/text2vec-base-chinese')
# model1 = SentenceTransformer('Alibaba-NLP/gte-multilingual-base', trust_remote_code=True)
# model2 = SentenceTransformer('HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v2')


class SentenceBert:
    def __init__(self):
        # self.model1 = SentenceTransformer('Alibaba-NLP/gte-multilingual-base', trust_remote_code=True)
        self.model1 = SentenceTransformer('shibing624/text2vec-base-chinese')
        self.model2 = SentenceTransformer('HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v2')

    def text2vector(self, model_type: int, text):
        if model_type == 1:
            return self.model1.encode(text, normalize_embeddings=True)
        elif model_type == 2:
            return self.model2.encode(text, normalize_embeddings=True)
        else:
            raise ValueError("Invalid model type")

    # 计算两个query的相似度
    def similarity(self, model_type: int, query1, query2):
        if model_type == 1:
            embedding1 = self.model1.encode(query1, convert_to_tensor=True)
            embedding2 = self.model1.encode(query2, convert_to_tensor=True)

            similarity = util.pytorch_cos_sim(embedding1, embedding2)
            print(similarity.item())
        elif model_type == 2:
            embedding1 = self.model2.encode(query1, convert_to_tensor=True)
            embedding2 = self.model2.encode(query2, convert_to_tensor=True)

            similarity = util.pytorch_cos_sim(embedding1, embedding2)
            print(similarity.item())
        else:
            raise ValueError("Invalid model type")



if __name__ == "__main__":
    if len(sys.argv) == 2:
        queryStr = sys.argv[1]
        sbert = SentenceBert()

        print("model1:")
        print(",".join(map(str, sbert.text2vector(1, queryStr))))
        print("model2:")
        print(",".join(map(str, sbert.text2vector(2, queryStr))))
        sys.exit(1)
    elif len(sys.argv) == 3:
        sbert = SentenceBert()
        print("model1:")
        sbert.similarity(1, sys.argv[1], sys.argv[2])
        print("model2:")
        sbert.similarity(2, sys.argv[1], sys.argv[2])

