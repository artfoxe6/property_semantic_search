from sentence_transformers import SentenceTransformer


class SentenceBert:

    @classmethod
    def text2vector(cls, model, text):
        embedding = model.encode(text, normalize_embeddings=True)
        return embedding
