import streamlit as st
import requests
import base64

# Apply CSS dynamically
st.markdown(
    """
    <style>
    /* Hide the Streamlit “Deploy” button */
    button[data-testid="stBaseButton-header"] {
        display: none;
    }

    /* Hide the Streamlit Settings menu (the gear icon) */
    span[data-testid="stMainMenu"] {
        display: none;
    }

    /* Hide uploader instruction text */
    div[data-testid="stFileUploaderDropzoneInstructions"]{
        display: none;
    }

    /* Keep the hamburger menu intact */
    </style>
    """,
    unsafe_allow_html=True
)

# Page Config
st.set_page_config(page_title="Mango AI", page_icon="🥭", layout="wide")

API_BASE = "http://localhost:8000"


# Sidebar: Training Section
with st.sidebar:
    st.title("📚 Train Your Bot")
    st.write("Upload a document **or** provide a URL (not both).")

    # ✅ Single file only
    uploaded_file = st.file_uploader("Upload a file (PDF, DOCX, TXT, max 5MB)", accept_multiple_files=False)
    url_input = st.text_input("Or enter a URL")

    # ✅ Validate file
    if uploaded_file:
        # Validate file size (max 5 MB)
        if uploaded_file.size > 5 * 1024 * 1024:
            st.error("❌ File size exceeds 5 MB. Please upload a smaller file.")
            uploaded_file = None  # Reset so it doesn't proceed
        else:
            # Validate file extension
            allowed_extensions = [".pdf", ".docx", ".txt"]
            file_name_lower = uploaded_file.name.lower()
            if not any(file_name_lower.endswith(ext) for ext in allowed_extensions):
                st.error("❌ Invalid file type. Please upload a PDF, DOCX, or TXT file.")
                uploaded_file = None

    # ✅ Enforce restriction: only one input allowed
    if uploaded_file and url_input:
        st.warning("⚠ Please provide either a URL or a file, not both.")
    elif st.button("Train Knowledge Base"):
        if uploaded_file or url_input:
            with st.spinner("Processing and creating vector database..."):
                files_data = {}

                # ✅ Convert uploaded file to base64 safely
                if uploaded_file:
                    try:
                        file_bytes = uploaded_file.read()
                        encoded = base64.b64encode(file_bytes).decode("ascii")
                        files_data[uploaded_file.name] = encoded
                    except Exception as e:
                        st.error(f"Failed to process file {uploaded_file.name}: {e}")

                payload = {"url": url_input if url_input else None, "files": files_data}

                try:
                    res = requests.post(f"{API_BASE}/train", json=payload)

                    if res.status_code == 200:
                        response = res.json()
                        # ✅ Save the active knowledge base info
                        if url_input:
                            st.session_state.active_kb = f"🌐 Knowledge Base: {url_input}"
                        elif uploaded_file:
                            st.session_state.active_kb = f"📄 Knowledge Base: {uploaded_file.name}"

                        st.success("✅ Knowledge base updated successfully!")
                        st.json(response)
                    else:
                        st.error(f"❌ Failed to train knowledge base. Status: {res.status_code}")
                        st.text(res.text)
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Please upload a file or provide a URL before training.")

# Main Chat UI
st.title("ChatB🤖t ⚡Powered by RAG")

# ✅ Show active knowledge base at the top of chat area
if "active_kb" in st.session_state:
    st.info(st.session_state.active_kb)
else:
    st.warning("No knowledge base trained yet. Please upload a document or provide a URL.")

# Initialize chat messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📂 Sources"):
                for idx, src in enumerate(msg["sources"], 1):
                    st.markdown(f"**Chunk {idx}:** {src}")

# User query input
if user_query := st.chat_input("Ask a question about your documents..."):
    # Show user message
    st.chat_message("user").markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Send query to API
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                res = requests.post(f"{API_BASE}/query", json={"question": user_query})
                if res.status_code == 200:
                    response_data = res.json()
                    answer = response_data.get("answer", "Sorry, I couldn't find an answer.")
                    sources = response_data.get("retrieved_chunks", [])
                else:
                    answer = "Error: Could not get response from API."
                    sources = []
            except Exception as e:
                answer = f"Error: {str(e)}"
                sources = []

            st.markdown(answer)

            # ✅ Show sources as expandable
            if sources:
                with st.expander("📂 Sources"):
                    for idx, src in enumerate(sources, 1):
                        st.markdown(f"**Chunk {idx}:** {src}")

            # ✅ Save to chat history
            st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
