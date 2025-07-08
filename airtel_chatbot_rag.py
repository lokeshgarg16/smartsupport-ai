import streamlit as st
import json
import os
import shutil
import redis
import pandas as pd
import pdfplumber
from deep_translator import GoogleTranslator
from langdetect import detect

# Vector DB and Embeddings
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# LangChain core
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import RedisChatMessageHistory

# Gemini API
import google.generativeai as genai

# --- Load environment variables from Streamlit Secrets ---
# --- Load environment variables from Streamlit Secrets ---
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    hf_token = st.secrets.get("HUGGINGFACEHUB_API_TOKEN", os.getenv("HUGGINGFACEHUB_API_TOKEN"))
    redis_url = st.secrets.get("REDIS_URL", "redis://localhost:6379")

    if not api_key or not hf_token:
        raise KeyError("One or more required secrets are missing")

    st.sidebar.success("✅ All secrets loaded successfully")
except KeyError as e:
    st.error(f"❌ Missing secret: {e}. Please add it in Streamlit > Settings > Secrets.")
    st.stop()

# --- Gemini Setup ---
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

# --- Redis Setup ---
try:
    r = redis.from_url(redis_url)
    r.ping()
    st.sidebar.success("✅ Connected to Redis")
except Exception as err:
    st.sidebar.error(f"❌ Redis connection failed: {err}")
    st.stop()

processed_key = "processed_files"

# --- Embedding and Vector Store Setup ---
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

UPLOAD_FOLDER = "./uploaded_backup"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Advanced Chunking ---
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", "!", "?", " ", ""]
)

# Function to load Vector Store
def load_vectorstore():
    return Chroma(
        collection_name="airtel_faqs",
        embedding_function=embedding_model
    )

vectorstore = load_vectorstore()

# Load base content
with open("faq_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)
docs = [Document(page_content=chunk, metadata={"source": "base_faq"}) for chunk in chunks]

# Load vectorstore with base data if not already
if not os.path.exists("./chroma_db/index") or not os.listdir("./chroma_db/index"):
    vectorstore.add_documents(docs)

# --- Streamlit UI ---
st.set_page_config(page_title="SmartSupport AI", page_icon="📺")
st.title("📺 SmartSupport AI")
st.caption("Ask in any language or upload your own Airtel support docs (PDF/CSV).")

# --- Chat history ---
chat_id = "airtel_session"
history = RedisChatMessageHistory(session_id=chat_id, url="redis://localhost:6379")
memory = ConversationBufferMemory(chat_memory=history, return_messages=True, memory_key="chat_history")

if st.button("Clear Chat"):
    history.clear()
    st.rerun()

if "chat_ended" not in st.session_state:
    st.session_state.chat_ended = False

if not st.session_state.chat_ended:
    if st.button("End Chat"):
        history.clear()
        for key in r.scan_iter("*"):
            r.delete(key)
        st.session_state.chat_ended = True
        st.success("Chat ended. You can start a new one below.")
else:
    if st.button("Start New Chat"):
        history.clear()
        st.session_state.chat_ended = False
        st.rerun()

if st.button("Reset All Files"):
    try:
        del vectorstore
        shutil.rmtree("./chroma_db", ignore_errors=True)
        os.makedirs("./chroma_db", exist_ok=True)
        vectorstore = load_vectorstore()
        vectorstore.add_documents(docs)
        r.delete(processed_key)
        shutil.rmtree(UPLOAD_FOLDER, ignore_errors=True)
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        for key in r.scan_iter("qa_cache:*"):
            r.delete(key)
        for key in r.scan_iter("questions_for:*"):
            r.delete(key)
        st.success("✅ All uploaded files and vector store reset!")
    except Exception as e:
        st.error(f"❌ Reset failed: {e}")

if st.session_state.get("chat_ended"):
    r.delete(processed_key)
    shutil.rmtree(UPLOAD_FOLDER, ignore_errors=True)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

processed_files = r.smembers(processed_key)
processed_file_names = [f.decode("utf-8") for f in processed_files]

st.subheader("📄 Uploaded Files")
if processed_file_names:
    for file in processed_file_names:
        col1, col2 = st.columns([4, 1])
        col1.markdown(f"- {file}")
        if col2.button("❌ Remove", key=f"remove_{file}"):

            for q_key in r.smembers(f"questions_for:{file}"):
                r.delete(f"qa_cache:{file}:{q_key.decode('utf-8')}")
            r.delete(f"questions_for:{file}")

            r.srem(processed_key, file)
            r.delete(f"chat_history_{file}")
            r.delete(chat_id)

            file_path = os.path.join(UPLOAD_FOLDER, file)
            if os.path.exists(file_path):
                os.remove(file_path)

            try:
                vectorstore._collection.delete(where={"source": file})
            except Exception as e:
                st.warning(f"⚠ Fallback to rebuild due to: {e}")
                shutil.rmtree("./chroma_db", ignore_errors=True)
                os.makedirs("./chroma_db", exist_ok=True)
                vectorstore = load_vectorstore()
                vectorstore.add_documents(docs)
                for f in r.smembers(processed_key):
                    f_name = f.decode("utf-8")
                    path = os.path.join(UPLOAD_FOLDER, f_name)
                    if os.path.exists(path):
                        with open(path, "rb") as up:
                            new_chunks = []
                            if path.endswith(".pdf"):
                                with pdfplumber.open(up) as pdf:
                                    for page_num, page in enumerate(pdf.pages):
                                        text = page.extract_text()
                                        if text:
                                            for i, chunk in enumerate(splitter.split_text(text)):
                                                new_chunks.append((chunk, {"source": f_name, "page": page_num+1, "chunk_id": i}))
                            elif path.endswith(".csv"):
                                df = pd.read_csv(up)
                                for idx, row in df.iterrows():
                                    if 'question' in df.columns and 'answer' in df.columns:
                                        qa = f"Q: {row['question']} A: {row['answer']}"
                                        new_chunks.append((qa, {"source": f_name, "row": idx}))
                                    else:
                                        new_chunks.append((str(row.to_dict()), {"source": f_name, "row": idx}))
                            if new_chunks:
                                vectorstore.add_documents([Document(page_content=chunk, metadata=meta) for chunk, meta in new_chunks])
            st.rerun()
else:
    st.info("No files uploaded yet.")

if len(processed_file_names) < 3:
    uploaded_files = st.file_uploader(
        f"📂 Upload PDF/CSV files ({3 - len(processed_file_names)} remaining)",
        type=["pdf", "csv"],
        accept_multiple_files=True
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name
            is_new_file = file_name not in processed_file_names

            if is_new_file and len(processed_file_names) >= 3:
                st.warning(f"🚫 Upload limit reached. '{file_name}' not uploaded.")
                continue

            with st.spinner(f"🔄 Processing '{file_name}'..."):
                file_path = os.path.join(UPLOAD_FOLDER, file_name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                new_chunks = []
                if uploaded_file.type == "application/pdf":
                    with pdfplumber.open(file_path) as pdf:
                        for page_num, page in enumerate(pdf.pages):
                            text = page.extract_text()
                            if text:
                                for i, chunk in enumerate(splitter.split_text(text)):
                                    new_chunks.append((chunk, {"source": file_name, "page": page_num+1, "chunk_id": i}))
                elif uploaded_file.type == "text/csv":
                    df = pd.read_csv(file_path)
                    for idx, row in df.iterrows():
                        if 'question' in df.columns and 'answer' in df.columns:
                            qa = f"Q: {row['question']} A: {row['answer']}"
                            new_chunks.append((qa, {"source": file_name, "row": idx}))
                        else:
                            new_chunks.append((str(row.to_dict()), {"source": file_name, "row": idx}))

                if new_chunks:
                    vectorstore.add_documents([Document(page_content=chunk, metadata=meta) for chunk, meta in new_chunks])
                    r.sadd(processed_key, file_name)
                    st.success(f"✅ '{file_name}' uploaded and added!")
                else:
                    st.warning(f"⚠ No readable content in '{file_name}'")
        st.rerun()
else:
    st.warning("📁 Maximum of 3 files uploaded. Please remove one to add more.")

# --- Show chat history ---
if not st.session_state.chat_ended:
    for msg in history.messages[-20:]:
        role = "🤑" if msg.type == "human" else "🧠"
        st.markdown(f"{role}:** {msg.content}")

    query = st.text_input("💬 Ask your question", placeholder="Ex: Which is best DTH?")
    if query:
        try:
            cached_answer = None
            source_file = None
            for file in processed_file_names:
                ans = r.get(f"qa_cache:{file}:{query.strip().lower()}")
                if ans:
                    cached_answer = ans
                    source_file = file
                    break

            if not cached_answer:
                ans = r.get(f"qa_cache:gemini_only:{query.strip().lower()}")
                if ans:
                    cached_answer = ans

            if cached_answer:
                st.markdown("### 🧠 Answer (from Redis Cache)")
                st.success(cached_answer.decode("utf-8"))
                history.add_user_message(query)
                history.add_ai_message(cached_answer.decode("utf-8"))
                st.stop()

            detected_lang = detect(query)
            translated_query = GoogleTranslator(source=detected_lang, target="en").translate(query)

            relevant_docs = vectorstore.similarity_search(translated_query, k=3)
            context = "\n\n".join([doc.page_content for doc in relevant_docs])

            prompt = f"""Answer the question based only on the information below.
If you don't know the answer, say \"Sorry! We do not have that information right now.\"

Context:
{context}

Question: {translated_query}
Answer:"""
            response = model.generate_content(prompt)
            english_answer = response.text

            if "Sorry! We do not have" in english_answer:
                fallback = model.generate_content("Answer from scratch: " + translated_query)
                english_answer = fallback.text

            translated_answer = GoogleTranslator(source="en", target=detected_lang).translate(english_answer)

            matched = False
            for file in processed_file_names:
                if any(file in doc.metadata.get("source", "") for doc in relevant_docs):
                    r.set(f"qa_cache:{file}:{query.strip().lower()}", translated_answer)
                    r.sadd(f"questions_for:{file}", query.strip().lower())
                    matched = True
                    break

            if not matched:
                r.set(f"qa_cache:gemini_only:{query.strip().lower()}", translated_answer)
                r.sadd("questions_for:gemini_only", query.strip().lower())

            history.add_user_message(query)
            history.add_ai_message(translated_answer)

            st.markdown("### 🧠 Answer")
            st.success(translated_answer)

            with st.expander("🔍 Retrieved Context"):
                st.text(context)

        except Exception as e:
            st.error(f"❌ Error: {e}")
