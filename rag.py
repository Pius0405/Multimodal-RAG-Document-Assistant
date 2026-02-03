from vector_stores import VectorStoreService
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
import config
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from prompt import SYSTEM_PROMPT
from file_history_store import get_history
import config


class RagService():
    def __init__(self):
        self.vector_service = VectorStoreService(
            embedding=HuggingFaceEmbeddings(model_name=config.embedding_model)
        )

        self.prompt_template=ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("system", "I provided the users chat history as follows"),
                MessagesPlaceholder("history"),
                ("user", "Please answer user query: {input}")
            ]
        )

        llm = HuggingFaceEndpoint(
            repo_id="Qwen/Qwen2.5-7B-Instruct",
            task="text-generation",
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.03,
            provider="auto",  
        )

        self.chat_model = ChatHuggingFace(llm=llm)

        self.chain = self.__get_chain()

    
    def __get_chain(self):
        retriever = self.vector_service.get_retriever()

        def format_document(docs):
            if not docs:
                return "No relavant information for reference"
            else:
                formatted_str=""
                for doc in docs:
                    formatted_str += f"Document chunk: {doc.page_content}\nDcoument metadata: {doc.metadata}"
                return formatted_str
        
        def format_for_retriever(value):
            return value["input"]

        def format_for_prompt_template(value):
            new_value = {}
            new_value["input"] = value["input"]["input"]
            new_value["context"] = value["context"]
            new_value["history"] = value["input"]["history"]
            return new_value


        chain = (
            {
                "input":RunnablePassthrough(),
                "context": RunnableLambda(format_for_retriever) | retriever | format_document
            } | RunnableLambda(format_for_prompt_template) | self.prompt_template | self.chat_model | StrOutputParser()
        )

        conversation_chain = RunnableWithMessageHistory(
            chain, 
            get_history,
            input_messages_key="input",
            history_messages_key="history"
        )

        return conversation_chain