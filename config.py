md5_path = "./md5.txt"
collection_name="policy_knowledge_base"
persist_directory="./chroma_db"
separators=["\n\n", "\n", ".", "?", "!", " "]
max_split_char_number=2000
similarity_threshold=3
embedding_model="all-MiniLM-L6-v2"
session_config = {
        "configurable": {
            "session_id":"user001"
        }
    }