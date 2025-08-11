# main.py
from typing import List
import chromadb
# from transformers import AutoTokenizer, AutoModel
import torch
import json
# from langchain.llms import LlamaCpp
import re
from underthesea import word_tokenize
import logging
import json
from pathlib import Path
from langchain.schema import Document
import os

from sentence_transformers import SentenceTransformer


import time
from datetime import datetime

from chromadb import Client, Documents, EmbeddingFunction, Embeddings

from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# class BGEM3EmbeddingFunction(EmbeddingFunction):
#     def __init__(
#         self, model_name="BAAI/bge-m3", batch_size=12, max_length=1024, use_fp16=True
#     ):
#         try:
#             from FlagEmbedding import BGEM3FlagModel

#             self._model = BGEM3FlagModel(model_name, use_fp16=use_fp16)
#             self._batch_size = batch_size
#             self._max_length = max_length
#         except ImportError:
#             raise ValueError(
#                 "The FlagEmbedding package is not installed. Please install it with "
#                 "pip install FlagEmbedding or uv add FlagEmbedding"
#             )

#     def __call__(self, input: Documents) -> Embeddings:
#         embeddings = self._model.encode(
#             input,
#             batch_size=self._batch_size,
#             max_length=self._max_length,
#         )["dense_vecs"]

#         return [e.tolist() for e in embeddings]


# Vietnamese Embedding Model
# class VietnameseEmbedding:
#     def __init__(self, model_name="BAAI/bge-m3"):
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.model = AutoModel.from_pretrained(model_name)
#         self.model.eval()
        
#     def encode(self, texts: List[str]) -> List[List[float]]:
#         """Encode Vietnamese texts to embeddings"""
#         embeddings = []
#         for text in texts:
#             inputs = self.tokenizer(text, return_tensors="pt", 
#                                   padding=True, truncation=True, max_length=512)
#             with torch.no_grad():
#                 outputs = self.model(**inputs)
#                 # Use mean pooling
#                 embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
#                 embeddings.append(embedding.tolist())
#         return embeddings

class VietnameseEmbedding:
    def __init__(self, model_name="BAAI/bge-m3"):
        # Load model từ sentence-transformers
        self.model = SentenceTransformer(model_name)
        self.model.eval()

    def encode(self, texts):
        """Encode Vietnamese texts to normalized embeddings"""
        return self.model.encode(texts, normalize_embeddings=True).tolist()



# RAG System
class VietnameseRAGSystem:
    def __init__(self):
        self.embedding_model = VietnameseEmbedding()
        self.chroma_client = chromadb.PersistentClient(path=r"D:\vitext2sql_vi\vitext2sql\pre\sql_ex_collection")
        self.collections = {}
        
        # # Initialize LLM (you can switch between models)
        # self.llm = self._init_llm()
        
    # def _init_llm(self):
    #     return None
    
    
    def setup_database(self, database_name: str,db_folder):
        """Setup database schema and embeddings"""
        # db_folder = os.path.join("db", database_name)
        self.chroma_client = chromadb.PersistentClient(path=os.path.join(db_folder, "db_chroma"))
        try:
            
            # collection_questions = None
            collection_db_des = None
            # try:
            #     with open(r"train.json", "r", encoding="utf-8") as f:
            #         data = json.load(f)
            # except FileNotFoundError:
            #     logger.error("File D:\\RAG_Vitext2sql\\train.json not found")
            #     raise ValueError("File train.json not found")
            # except json.JSONDecodeError:
            #     logger.error("Invalid JSON format in train.json")
            #     raise ValueError("Invalid JSON format in train.json")
            # with open(r"D:\vitext2sql_vi\vitext2sql\pre\train.json", "r", encoding="utf-8") as f:
            #     data = json.load(f)

            # def remove_punctuation(text):
            #     text = text.lower()
            #     # Xóa dấu phẩy, chấm, chấm phẩy, chấm than, hỏi
            #     text = re.sub(r"[.,;!?]", " ", text)
            #     # Nhưng giữ lại dấu . trong tên người hoặc số liệu
            #     text = re.sub(r"\s+", " ", text).strip()
            #     return text
            
            # # Tạo danh sách documents
            # documents = []
            # with open(r"D:\vitext2sql_vi\vitext2sql\pre\tables.json", "r", encoding="utf-8") as f:
            #     schemas = json.load(f)

            

            # # Từ khóa SQL không cần quote
            # sql_keywords = {
            #     "select", "from", "where", "join", "on", "as", "and", "or",
            #     "=", ">", "<", ">=", "<=", "<>", "!=", "*"
            # }
            # def quote_token(token: str) -> str:
            #     """Quote tên bảng/cột nếu cần"""
            #     stripped = token.strip()
            #     lower = stripped.lower()

            #     # Không quote nếu là từ khóa SQL hoặc số hoặc đã có quote
            #     if (lower in sql_keywords
            #         or re.match(r'^[0-9]+$', stripped)
            #         or stripped.startswith('"')
            #         or stripped.startswith("'")):
            #         return token

            #     # Nếu có dấu .
            #     if "." in stripped:
            #         parts = stripped.split(".")
            #         quoted_parts = []
            #         for part in parts:
            #             if part in table_set or part in column_set:
            #                 quoted_parts.append(f'"{part}"')
            #             else:
            #                 quoted_parts.append(part)
            #         return ".".join(quoted_parts)

            #     # Không có dấu . thì kiểm tra trực tiếp
            #     if stripped in table_set or stripped in column_set:
            #         return f'"{stripped}"'

            #     return token

            # for item in data:
            #     question = remove_punctuation(item.get("question", "")).strip()
            #     # query = item.get("query", "").strip()
            #     db_id = item.get("db_id", "").strip()

            #     # Lấy schema của academic
            #     academic_schemas = next(s for s in schemas if s["db_id"] == item["db_id"])

            #     # Tập hợp tên bảng và tên cột
            #     table_set = set(academic_schemas["table_names_original"])
            #     column_set = set(col[1] for col in academic_schemas["column_names_original"])
            #     quoted_tokens = [quote_token(tok) for tok in item["query_toks"]]
            #     token_based_query = " ".join(quoted_tokens).strip()

            #     if question and token_based_query:
            #         doc = {
            #             "text": question,
            #             "metadata": {
            #                 "sql_text": token_based_query,
            #                 "db_id": db_id,
            #             }
            #         }
            #         documents.append(doc)

            # if "sql_query_questions" in [col.name for col in self.chroma_client.list_collections()]:
            #     self.chroma_client.delete_collection("sql_query_questions")
            # # Tạo collection cho question-query
            # collection_questions = self.chroma_client.create_collection(
            #     name="sql_query_questions",metadata={"hnsw:space": "ip"})

            # texts = [doc["text"] for doc in documents]
            # metadatas = [doc["metadata"] for doc in documents]
            # ids = [f"sql_qa_{i}" for i in range(len(texts))]
            # embeddings = self.embedding_model.encode(texts)
            # print("texts:"+  texts[0])
            # BATCH_SIZE = 100  # nhỏ hơn 166
            # logger.info(f"Processing {len(texts)} documents for  collection")
            # for i in range(0, len(texts), BATCH_SIZE):
            #     collection_questions.add(
            #         documents=texts[i:i + BATCH_SIZE],
            #         metadatas=metadatas[i:i + BATCH_SIZE],
            #         ids=ids[i:i + BATCH_SIZE],
            #         embeddings=embeddings[i:i + BATCH_SIZE]
            #     )
  

            input_file = os.path.join(db_folder,"db_des", "db_des.txt")
            output_dir = os.path.join(db_folder,"db_des","chunks")

            # Tạo thư mục lưu chunk nếu chưa có
            os.makedirs(output_dir, exist_ok=True)

            # Đọc file txt
            with open(input_file, "r", encoding="utf-8") as f:
                text = f.read()

            # Chia theo dòng
            lines = text.splitlines()

            # Hàm chuẩn hóa (bỏ dấu câu, viết thường)
            def remove_punctuation(text):
                text = text.lower()
                text = re.sub(r"[.,;!?]", " ", text)
                text = re.sub(r"\s+", " ", text).strip()
                return text

            # Lọc và xử lý từng dòng
            lines = [remove_punctuation(line) for line in lines if line.strip()]

            # Ghép 3 dòng thành 1 chunk
            chunk_size = 3
            chunks = [
                "\n".join(lines[i:i+chunk_size]) 
                for i in range(0, len(lines), chunk_size)
            ]

            # Lưu từng chunk thành file txt riêng
            for idx, chunk in enumerate(chunks, start=1):
                file_path = os.path.join(output_dir, f"chunk_{idx}.txt")
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(chunk)

            def load_documents_from_txt_folder1(folder_path: str) -> list[Document]:
                documents = []
                folder = Path(folder_path)

                for file_path in folder.glob("*.txt"):
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read().strip()

                    if content:
                        document = Document(
                            page_content=content,
                            metadata={
                                "source": file_path.stem,  # tên file không có .txt
                                "source_type": "db_des"
                            }
                        )
                        documents.append(document)

                print(f"Tải được {len(documents)} document từ thư mục {folder_path}")
                return documents
            
            collection_db_des = self.chroma_client.create_collection(
                name="db_des")
            processed_docs = load_documents_from_txt_folder1(output_dir)
            if not processed_docs:
                logger.error(f"No documents found in folder {output_dir}")
                raise ValueError("No documents found for sql_tutorial collection")
            texts_db_des = [doc.page_content for doc in processed_docs]
            metadatas_db_des = [doc.metadata for doc in processed_docs]
            ids_db_des = [f"sql_tutorial_{i}" for i in range(len(texts_db_des))]

            print("texts_db_des:"+  texts_db_des[0])

            embeddings_db_des = self.embedding_model.encode(texts_db_des)
            BATCH_SIZE = 100  # nhỏ hơn 166
            logger.info(f"Processing {len(texts_db_des)} documents for db_des collection")
            for i in range(0, len(texts_db_des), BATCH_SIZE):
                collection_db_des.add(
                    documents=texts_db_des[i:i + BATCH_SIZE],
                    metadatas=metadatas_db_des[i:i + BATCH_SIZE],
                    ids=ids_db_des[i:i + BATCH_SIZE],
                    embeddings=embeddings_db_des[i:i + BATCH_SIZE]
                )

            self.collections[database_name] = {
            "sql_tutorial": collection_db_des
            # "sql_query_questions": collection_questions
        }
            logger.info(f"Setup completed for database: {database_name}")
            
        except Exception as e:
            logger.error(f"Error setting up database: {e}")
            raise
# class IndexProcessor:
#     def __init__(self, chroma_client: chromadb.ClientAPI, emb_func: EmbeddingFunction):
#         self.client = chroma_client
#         self.embedding_function = EmbeddingFunction

#         self.collections = []

#     @staticmethod
#     def read_datasource(path) -> list[str]:
#         def rm_punc(text):
#             text = text.lower()
#             # Xóa dấu phẩy, chấm, chấm phẩy, chấm than, hỏi
#             text = re.sub(r"[.,;!?]", " ", text)
#             # Nhưng giữ lại dấu . trong tên người hoặc số liệu
#             text = re.sub(r"\s+", " ", text).strip()
#             return text

#         with open(path, "r") as f:
#             data = json.load(f)

#         docs = []
#         for item in data:
#             question = rm_punc(item.get("question", "")).strip()
#             query = item.get("query", "").strip()
#             db_id = item.get("db_id", "").strip()
#             query_toks = item.get("query_toks", "").strip()
#             if question and query:
#                 doc = {
#                     "text": question,
#                     "metadata": {
#                         "sql_text": query,
#                         "db_id": db_id,
#                         "query_toks": query_toks,
#                     },
#                 }
#                 docs.append(doc)

#         return docs

#     def add_exemplars(
#         self,
#         datasource_path: str,
#         collection_name: str = "exemplars",
#         batch_size: int = 100,
#     ):
#         """Method for ingesting question-sql pairs"""

#         documents = self.read_datasource(datasource_path)

#         now = datetime.now()
#         collection = self.client.create_collection(
#             name=f"{collection_name}_{now.strftime('%H_%M_%S_%Y_%m_%d')}",
#             embedding_function=self.embedding_function,
#         )

#         texts = [doc["text"] for doc in documents]
#         metadatas = [doc["metadata"] for doc in documents]
#         ids = [f"exemplar_{i}" for i in range(texts)]

#         for i in range(0, len(texts), batch_size):
#             collection.add(
#                 documents=texts[i : i + batch_size],
#                 metadatas=metadatas[i : i + batch_size],
#                 ids=ids[i : i + batch_size],
#             )

#         self.collections[f"{collection_name}_{now.strftime('%H_%M_%S_%Y_%m_%d')}"] = (
#             collection
#         )

if __name__ == "__main__":
    
    database_name = "baseball_1"
    rag_system = VietnameseRAGSystem()
    rag_system.setup_database(database_name)
   
