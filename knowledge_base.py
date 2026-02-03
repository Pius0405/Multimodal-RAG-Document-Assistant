import os
import config
import hashlib
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime


def check_md5(md5_str):
    if not os.path.exists(config.md5_path):
        open(config.md5_path, "w", encoding="utf-8").close()
        return False
    else:
        f = open(config.md5_path, "r", encoding="utf-8")
        for line in f.readlines():
            line = line.strip()
            if (line == md5_str):
                f.close()
                return True
        f.close()
        return False


def save_md5(md5_str):
    with open(config.md5_path, "a", encoding="utf-8") as f:
        f.write(md5_str + "\n")


def get_string_md5(input_str, encoding="utf-8"):
    # Convert string to bytes  
    str_bytes = input_str.encode(encoding=encoding)
    # Create md5 object
    md5_obj = hashlib.md5()
    md5_obj.update(str_bytes)
    md5_hex = md5_obj.hexdigest()
    return md5_hex


class KnowledgeBaseService():
    def __init__(self):
        os.makedirs(config.persist_directory, exist_ok=True)
        self.chroma = Chroma(
            collection_name=config.collection_name,
            embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
            persist_directory=config.persist_directory,
        )


        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            separators=config.separators,
            length_function=len
        )

    def upload_by_str(self, data, filename):
        md5_hex = get_string_md5(data)
        if check_md5(md5_hex):
            return "Content already exists in knowledge base"
        
        if len(data) > config.max_split_char_number:
            knowledge_chunks = self.splitter.split_text(data)
        else:
            knowledge_chunks = [data]
        
        metadata = {
            "source": filename,
            "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operator": "Pius"
        }

        self.chroma.add_texts(
            knowledge_chunks,
            metadatas=[metadata for _ in knowledge_chunks],
            ids = ["id"+str(i) for i in range(1, len(knowledge_chunks)+1)]
        )

        save_md5(md5_hex)

        return "Content successfully stored in vector database"
    



