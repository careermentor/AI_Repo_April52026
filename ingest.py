import os
import pickle
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def extract_text_from_pdf(pdf_path):
    print(f"Reading {pdf_path}...")
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=800, overlap=100):
    print("Chunking text...")
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += (chunk_size - overlap)
    return chunks

def create_embeddings(chunks):
    print("Loading embedding model and generating embeddings...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks)
    return model, embeddings

def save_vector_store(embeddings, chunks):
    print("Saving vector store to disk...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    
    # Save index and chunks (metadata)
    faiss.write_index(index, "index.faiss")
    with open("metadata.pkl", "wb") as f:
        pickle.dump(chunks, f)
    print("Saved index.faiss and metadata.pkl.")

def main():
    pdf_path = "Articles.pdf"
    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} not found.")
        return
    
    raw_text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(raw_text)
    model, embeddings = create_embeddings(chunks)
    save_vector_store(embeddings, chunks)

if __name__ == "__main__":
    main()
