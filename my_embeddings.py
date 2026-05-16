from google import genai
from langchain_core.embeddings import Embeddings


class BentechEmbeddings(Embeddings):
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)
        self.model_id = "gemini-embedding-2" 

    def embed_documents(self, texts):
        embeddings_list = []
        for text in texts:
            response = self.client.models.embed_content(
                model=self.model_id, contents=text
            )
            embeddings_list.append(response.embeddings[0].values)
        return embeddings_list

    def embed_query(self, text):
        response = self.client.models.embed_content(
            model=self.model_id, contents=text
        )
        return response.embeddings[0].values