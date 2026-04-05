import streamlit as st
from qa_engine import QAEngine

# Configuration
st.set_page_config(page_title="PDF Question Answering", layout="centered")

# Load engine once (caching)
@st.cache_resource
def load_engine():
    return QAEngine()

def main():
    st.title("📘 Articles QA Assistant")
    st.markdown("Ask questions based on the pre-loaded `Articles.pdf` document.")

    # Sidebar info
    with st.sidebar:
        st.info("Knowledge Base: `Articles.pdf`")
        st.info("Embedding: `all-MiniLM-L6-v2`")
        st.info("LLM: `flan-t5-base` (Running locally)")

    # Engine initialization
    try:
        engine = load_engine()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return

    # Question Input
    user_question = st.text_input("Enter your question here:")

    if st.button("Get Answer"):
        if user_question.strip():
            with st.spinner("Searching for context and generating answer..."):
                answer, chunks = engine.get_answer(user_question)
                
                # Display Answer
                st.subheader("Answer:")
                st.write(answer)
                
                # Display Source Context (Optional, for transparency)
                with st.expander("Show source context"):
                    for i, chunk in enumerate(chunks):
                        st.write(f"**Source {i+1}:**")
                        st.markdown(f"> {chunk}")
        else:
            st.warning("Please enter a question.")

if __name__ == "__main__":
    main()
