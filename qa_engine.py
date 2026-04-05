import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Constants
INDEX_FILE = "index.faiss"
METADATA_FILE = "metadata.pkl"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
LLM_MODEL_NAME = 'google/flan-t5-base'

class QAEngine:
    def __init__(self):
        print("Loading models...")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

        # Load LLM components directly
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
        self.llm = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL_NAME)

        print("Loading vector index...")
        self.index = faiss.read_index(INDEX_FILE)
        with open(METADATA_FILE, "rb") as f:
            self.metadata = pickle.load(f)

    def search(self, query, top_k=3):
        query_embedding = self.embedding_model.encode([query])
        distances, indices = self.index.search(np.array(query_embedding).astype('float32'), top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                results.append({
                    "content": self.metadata[idx],
                    "score": distances[0][i]
                })
        return results

    def get_answer(self, question):
        relevant_chunks = self.search(question)

        if not relevant_chunks or relevant_chunks[0]['score'] > 2.0: # Increased threshold slightly
            return "Answer not found in document.", []

        context = "\n\n".join([res['content'] for res in relevant_chunks])

        prompt = f"""You are a helpful assistant.
Answer ONLY using the context provided.

Context:
{context}

Question:
{question}

Answer:"""

        print("Generating answer...")
        # Tokenize and generate
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        outputs = self.llm.generate(**inputs, max_new_tokens=200)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return answer, [res['content'] for res in relevant_chunks]


if __name__ == "__main__":
    # Test
    engine = QAEngine()
    question = "What is the main topic of the document?"
    ans, context = engine.get_answer(question)
    print(f"\nQ: {question}\nA: {ans}")
