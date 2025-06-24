from sentence_transformers import SentenceTransformer


class SentenceBert:

    @classmethod
    def text2vector(cls, text):
        model = SentenceTransformer('shibing624/text2vec-base-chinese')
        embedding = model.encode(text, normalize_embeddings=True)
        return embedding
