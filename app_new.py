import streamlit as st
import pickle
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline
import torch
import os

# CONFIGURATION: UPGRADED OPEN SOURCE MODELS
INDEX_PATH = "index.faiss"
METADATA_PATH = "metadata.pkl"
EMBEDDING_MODEL_NAME = 'BAAI/bge-base-en-v1.5'
RERANK_MODEL_NAME = 'BAAI/bge-reranker-base'
LLM_MODEL_NAME = "google/flan-t5-large" # 3x larger than 'base' for better reasoning

# Page styling
st.set_page_config(page_title="RAG Chat Assistant", page_icon="🚀")
st.title("🚀 Conversational Assistant")


@st.cache_resource
def load_resources():
   
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    rerank_model = CrossEncoder(RERANK_MODEL_NAME)
    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, 'rb') as f:
        metadata_list = pickle.load(f)
    
    device = 0 if torch.cuda.is_available() else -1
    llm_pipeline = pipeline("text2text-generation", model=LLM_MODEL_NAME, device=device)
    
    return embed_model, rerank_model, index, metadata_list, llm_pipeline

try:
    embed_model, rerank_model, index, metadata_list, llm_pipeline = load_resources()
except Exception as e:
    st.error(f"Error loading resources: {e}")
    st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar Metadata Filtering
st.sidebar.header("🔍 Metadata Filtering")
all_pages = sorted(list(set(m['page_number'] for m in metadata_list)))
selected_pages = st.sidebar.multiselect("Select Pages to Search (Optional):", options=all_pages, default=[])

if metadata_list:
    sample_meta = metadata_list[0]
    st.sidebar.info(f"File: {sample_meta['file_name']}\nUpdated: {sample_meta['last_updated']}")

if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def rewrite_query(question, history):
    if not history:
        # BGE models work best with a query instruction
        return f"Represent this sentence for searching relevant passages: {question}"
    
    history_context = ""
    for msg in history[-2:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_context += f"{role}: {msg['content'][:150]}\n"
        
    rewrite_prompt = f"""Instruction: Rephrase the 'New Question' into a standalone search query that includes specific subjects mentioned in the 'Chat History'.

Example 1:
Chat History:
User: What is Article 1?
Assistant: It says all humans are born free and equal.
New Question: Tell me more about it.
Standalone Query: Represent this sentence for searching relevant passages: Detailed information about Article 1 regarding humans being born free.

Current Task:
Chat History:
{history_context}
New Question: {question}
Standalone Query: Represent this sentence for searching relevant passages:"""
    
    rewritten = llm_pipeline(rewrite_prompt, max_length=64, truncation=True)
    query = rewritten[0]['generated_text'].strip()
    query = query.replace("Standalone Query:", "").strip()
    
    return query if query else f"Represent this sentence for searching relevant passages: {question}"

# Question processing logic
def get_answer(question, history, page_filter):
    search_query = rewrite_query(question, history)
    search_query = search_query.strip().strip('"').strip("'")
    
    # Step 1: Initial Retrieval
    question_embedding = embed_model.encode([search_query], normalize_embeddings=True)
    distances, indices = index.search(question_embedding, k=20)
    
    initial_matches = []
    for idx in indices[0]:
        if idx < len(metadata_list):
            meta = metadata_list[idx]
            if not page_filter or meta['page_number'] in page_filter:
                initial_matches.append(meta)
    
    if not initial_matches:
        return "I'm sorry, I couldn't find any chunks matching that filter.", [], search_query
    
    # Step 2: Reranking Stage
    pairs = [[search_query, m['text']] for m in initial_matches[:10]]
    scores = rerank_model.predict(pairs)
    reranked_results = sorted(zip(scores, initial_matches[:10]), key=lambda x: x[0], reverse=True)
    
    # Keep the top 3 best chunks
    top_results = [res for score, res in reranked_results[:3]]
    
    # Step 3: Generation
    context_text = "\n".join([r['text'] for r in top_results])
    
    history_text = ""
    for msg in history[-2:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"
    
    # INVERTED PROMPT: Context first, then Instruction, then Question.
    # This helps models focus on the question as the final command.
    prompt = f"""Use the following Context to answer the User's Question.
    
Context:
{context_text}

Chat History:
{history_text}

Instruction: Provide a detailed but focused answer. ONLY mention the articles relevant to the question. Do not include unrelated articles.

User Question: {question}
Assistant Answer:"""
    
    response = llm_pipeline(prompt, max_length=256, truncation=True, do_sample=False)
    answer = response[0]['generated_text'].strip()
    
    # POST-PROCESSING: Clean up if the model hallucinates prompt headers
    for stop_word in ["User Question:", "Context:", "Chat History:", "Assistant Answer:"]:
        if stop_word in answer:
            answer = answer.split(stop_word)[0].strip()
            
    return answer, top_results, search_query

# Chat input
if user_question := st.chat_input("Ask a question about the PDF..."):
    with st.chat_message("user"):
        st.markdown(user_question)
    
    st.session_state.messages.append({"role": "user", "content": user_question})
    
    with st.chat_message("assistant"):
        with st.spinner("Analyzing document with high precision..."):
            answer, source_metas, search_query = get_answer(user_question, st.session_state.messages[:-1], selected_pages)
            st.markdown(answer)
            
            with st.expander("Debug: Pro RAG Info"):
                st.write(f"**Retrieval Model:** {EMBEDDING_MODEL_NAME}")
                st.write(f"**Reranker Model:** {RERANK_MODEL_NAME}")
                st.write(f"**Rewritten Query:** {search_query}")
                if source_metas:
                    for i, meta in enumerate(source_metas):
                        st.info(f"Source {i+1} - Page {meta['page_number']}:\n{meta['text']}")
    
    st.session_state.messages.append({"role": "assistant", "content": answer})



    #write a java program to reverse a string without using built in functionns.
    #Here is a simple Java program that reverses a string without using built-in functions:
 
        
        public static void main(String[] args) {
            String original = "Hello World";
            String reversed = reverseString(original);
            
            System.out.println("Original: " + original);
            System.out.println("Reversed: " + reversed);
        }
    }







