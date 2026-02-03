import streamlit as st
from rag import RagService
import config
from dotenv import load_dotenv
from knowledge_base import KnowledgeBaseService
import time
from image_captioner import ImageCaptionerHF
from pdf_processor import PDFProcessor
from file_history_store import get_history
from langchain_core.messages import HumanMessage, AIMessage

st.title("Multimodal RAG Chatbot")
st.divider()

if "application_started" not in st.session_state:
    st.session_state["application_started"] = True
    load_dotenv()

with st.sidebar:
    st.title("Update Knowledge Base")

    uploader_file = st.file_uploader(
        "Please upload a file here",
        type="pdf",
        accept_multiple_files=False,
    )

    if "service" not in st.session_state:
        st.session_state["service"] = KnowledgeBaseService()

    if uploader_file is not None:
        file_name = uploader_file.name
        file_type = uploader_file.type  
        file_size = uploader_file.size

        st.subheader(f"{file_name}")
        st.write(f"File type: {file_type} | File size: {file_size}")

        captioner = ImageCaptionerHF()
        processor = PDFProcessor(uploader_file, captioner)

        all_text_chunks = processor.process()

        # Combine everything into one string (or handle as separate chunks)
        combined_text = "\n\n".join([chunk["content"] for chunk in all_text_chunks])


        with st.spinner("Saving document to vector database"):
            time.sleep(1)
            result = st.session_state["service"].upload_by_str(combined_text, file_name)
            st.write(result)

if  "rag" not in st.session_state:
    st.session_state["rag"] = RagService()

if "message" not in st.session_state:

    session_id = config.session_config["configurable"]["session_id"]
        
    # Load chat history from file
    history_store = get_history(session_id)
    file_messages = history_store.messages
        
    # Convert LangChain messages to display format
    if file_messages:
        st.session_state["message"] = []
        for msg in file_messages:
            if isinstance(msg, HumanMessage):
                st.session_state["message"].append({
                    "role": "user",
                    "content": msg.content
                })
            elif isinstance(msg, AIMessage):
                st.session_state["message"].append({
                    "role": "assistant",
                    "content": msg.content
                })
    else:
        st.session_state["message"] = [{"role":"assistant", "content":"Hello, how can I assist you?"}]

for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])

prompt = st.chat_input()

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role":"user", "content":prompt})

    with st.spinner("Thinking..."):
        res = st.session_state["rag"].chain.invoke({"input":prompt}, config.session_config)
        st.chat_message("assistant").write(res)
        st.session_state["message"].append({"role":"assistant", "content":res})


