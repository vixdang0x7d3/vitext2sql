import chromadb
from typing import List, Dict
from transformers import AutoTokenizer, AutoModel
import torch
import logging
import os
from underthesea import ner
import time
import logging
# import py_vncorenlp

from sentence_transformers import SentenceTransformer


# Tắt telemetry của ChromaDB
# ===== 1. Cấu hình logging và tắt telemetry của ChromaDB =====
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ["CHROMADB_TELEMETRY_ENABLED"] = "False"
# old_cwd = os.getcwd()
# print(old_cwd)
# model = py_vncorenlp.VnCoreNLP(
#     annotators=["wseg", "pos", "ner", "parse"],
#     save_dir=r"D:\RAG_Vitext2sql\vncorenlp"
#     )
# print(os.getcwd())
# os.chdir(old_cwd)
# print(os.getcwd())
# ===== 2. Load Vietnamese Embedding Model (BGE-M3) =====
class VietnameseEmbedding:
    def __init__(self, model_name="BAAI/bge-m3"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        """Encode Vietnamese texts to normalized embeddings"""
        return self.model.encode(texts, normalize_embeddings=True).tolist()
# class VietnameseEmbedding:
#     def __init__(self, model_name="BAAI/bge-m3"):
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name, force_download=False)
#         self.model = AutoModel.from_pretrained(model_name, force_download=False)
#         self.model.eval()

#     def encode(self, texts: List[str]) -> List[List[float]]:
#         """Encode Vietnamese texts to embeddings"""
#         embeddings = []
#         for text in texts:
#             inputs = self.tokenizer(
#                 text, return_tensors="pt", padding=True, truncation=True, max_length=512
#             )
#             with torch.no_grad():
#                 outputs = self.model(**inputs)
#                 embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
#                 embeddings.append(embedding.tolist())
#         return embeddings


embedding_model = VietnameseEmbedding()


# def retrieve_from_collections(db_des: bool,question: str, database_name: str, top_k: int = 8) -> List[Dict]:
def retrieve_from_collections(db_folder,db_path,
    db_des: bool, question: str, database_name: str, top_k: int = 12,log_callback=None
) -> List[Dict]:
    """
    Retrieve relevant documents from sql_tutorial and sql_query_questions collections.

    Args:
        question (str): Câu hỏi cần truy vấn.
        database_name (str): Tên database (dùng để tổ chức collections).
        persist_dir (str): Thư mục lưu trữ ChromaDB.
        top_k (int): Số lượng tài liệu tối đa trả về.

    Returns:
        List[Dict]: Danh sách các tài liệu liên quan, mỗi tài liệu chứa id, type, content, distance, metadata.
    """
    logs = []

    def log(msg):
        if log_callback:
            log_callback(msg)
        else:
            logs.append(msg)
    try:
        # Kết nối tới ChromaDB
        
        chroma_client_sql_ex = chromadb.PersistentClient(path=r"pre\sql_ex_collection")
        # print(chroma_client_sql_ex.list_collections())
        # Lấy collections
        collection_questions = None
        collection_tutorial = None
        try:
            collection_questions = chroma_client_sql_ex.get_collection(
                name="sql_query_questions"
            )
            logger.info("Loaded sql_query_questions collection")
            log("Loaded sql_query_questions collection")
        except Exception as e:
            logger.warning(f"Could not load sql_query_questions collection: {e}")
            log(f"Could not load sql_query_questions collection: {e}")

        # if not collection_questions and not collection_tutorial:
        #     logger.error("Both collections could not be loaded")
        #     return []

        # Mã hóa câu hỏi
        question_ner = mask_entities(question)
        query_embedding_ex = embedding_model.encode([question_ner])
        retrieved_elements = []

        # Truy vấn sql_query_questions
        if collection_questions:
            results = collection_questions.query(
                query_embeddings=query_embedding_ex,
                n_results=top_k // 2,
                include=["documents", "metadatas", "distances"],  # Loại bỏ 'ids'
            )
            if results["documents"] and len(results["documents"][0]) > 0:
                for i, doc in enumerate(results["documents"][0]):
                    # Lấy id từ results['ids'] (tự động trả về trong chromadb 0.4.15)
                    doc_id = (
                        results.get("ids", [[]])[0][i]
                        if "ids" in results and len(results["ids"][0]) > i
                        else f"unknown_{i}"
                    )
                    retrieved_elements.append(
                        {
                            "id": doc_id,
                            "type": "vector_similarity_questions",
                            "content": doc,
                            "distance": results["distances"][0][i],
                            "metadata": results["metadatas"][0][i],
                        }
                    )
            else:
                log("No results found in sql_query_questions collection")
                logger.warning("No results found in sql_query_questions collection")
        query_embedding_db_des = embedding_model.encode([question])

        if db_des:
            # Truy vấn db des
            # db_folder = os.path.join("pre/db", database_name)

            chroma_client_db_des = chromadb.PersistentClient(
                path=os.path.join(db_folder, "db_chroma")
            )
            try:
                collection_tutorial = chroma_client_db_des.get_collection(name="db_des")
                logger.info("Loaded sql_tutorial collection")
                log("Loaded sql_tutorial collection")
            except Exception as e:
                log(f"Could not load sql_tutorial collection: {e}")
                logger.warning(f"Could not load sql_tutorial collection: {e}")

            if collection_tutorial:
                results = collection_tutorial.query(
                    query_embeddings=query_embedding_db_des,
                    n_results=top_k // 2,
                    include=["documents", "metadatas", "distances"],  # Loại bỏ 'ids'
                )
                if results["documents"] and len(results["documents"][0]) > 0:
                    for i, doc in enumerate(results["documents"][0]):
                        # Lấy id từ results['ids']
                        doc_id = (
                            results.get("ids", [[]])[0][i]
                            if "ids" in results and len(results["ids"][0]) > i
                            else f"unknown_{i}"
                        )
                        retrieved_elements.append(
                            {
                                "id": doc_id,
                                "type": "vector_similarity_tutorial",
                                "content": doc,
                                "distance": results["distances"][0][i],
                                "metadata": results["metadatas"][0][i],
                            }
                        )
                else:
                    log("No results found in database description collection")
                    logger.warning("No results found in database description collection")

        # Loại bỏ trùng lặp và giới hạn top_k
        seen_ids = set()
        unique_elements = []
        for elem in retrieved_elements:
            if elem["id"] not in seen_ids:
                unique_elements.append(elem)
                seen_ids.add(elem["id"])

        return unique_elements[:top_k],"\n".join(logs)

    except Exception as e:
        logger.error(f"Error retrieving from collections: {e}")
        return []


def create_prompt_context(
    db_des: bool, retrieved_results: List[Dict]
) -> tuple[str, str]:
    """
    Tạo context string ngắn gọn để đưa trực tiếp vào prompt cho LLM.

    Args:
        retrieved_results (List[Dict]): Kết quả từ hàm retrieve_from_collections

    Returns:
        str: Context string để đưa vào prompt
    """
    sql_ex_context_parts = []
    db_des_context_parts = []
    # SQL Examples
    sql_examples = [
        r for r in retrieved_results if r["type"] == "vector_similarity_questions"
    ]
    if sql_examples:
        sql_ex_context_parts.append("**Similar SQL Examples:**")
        for i, ex in enumerate(sql_examples[:4], 1):
            sql = ex["metadata"].get("sql_text", "")
            db = ex["metadata"].get("db_id", "")
            sql_ex_context_parts.append(f"{i}. Question: {ex['content']}")
            sql_ex_context_parts.append(f"   SQL: {sql}")
            sql_ex_context_parts.append(f"   Database: {db}")
    if db_des:
        # db description
        tutorials = [
            r for r in retrieved_results if r["type"] == "vector_similarity_tutorial"
        ]
        if tutorials:
            db_des_context_parts.append("\n**Relevant DB description Concepts:**")
            for i, tut in enumerate(tutorials[:6], 1):
                content = tut["content"]
                db_des_context_parts.append(f"{i}. {content}")

    return "\n".join(db_des_context_parts), "\n".join(sql_ex_context_parts)


def save_prompt_context(db_des:bool ,db_folder,db_path,
    results: List[Dict],
    output_sql_ex_file: str = "sql_ex_context",
    output_db_des_file: str = "db_des_context",
    id: int = int(time.time()),
    db_name: str = "",log_callback=None
) -> None:
    """
    Lưu context ngắn gọn để đưa vào prompt.


    Args:
        retrieved_results (List[Dict]): Kết quả từ hàm retrieve_from_collections
        output_file (str): Tên file output
    """

    # logs = []

    # def log(msg):
    #     if log_callback:
    #         log_callback(msg)
    #     else:
    #         logs.append(msg)
    db_des_context, sql_ex_context = create_prompt_context(db_des, results)

    # db_folder = os.path.join("pre/db", db_name)
    external_knowledge = os.path.join(db_folder, "external_knowledge")
    os.makedirs(external_knowledge, exist_ok=True)
    output_sql_ex_file = os.path.join(external_knowledge, output_sql_ex_file)
    os.makedirs(output_sql_ex_file, exist_ok=True)
    output_db_des_file = os.path.join(external_knowledge, output_db_des_file)
    os.makedirs(output_db_des_file, exist_ok=True)
    try:
        with open(
            os.path.join(output_sql_ex_file, str(id) + ".txt"), "w", encoding="utf-8"
        ) as f:
            f.write(sql_ex_context)
        print(f" Prompt context đã được lưu vào file: {output_sql_ex_file}")
        if db_des_context:
            with open(
                os.path.join(output_db_des_file, str(id) + ".txt"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write(db_des_context)
            print(f" Prompt context đã được lưu vào file: {output_db_des_file}")
    except Exception as e:
        print(f"Lỗi khi lưu prompt context: {e}")


def mask_entities(question: str) -> str:
    entities = ner(question)  # Trả về [(word, pos, chunk, ner_label), ...]
    # print(entities)
    masked_words = []

    for token, pos, chunk, ner_label in entities:
        if ner_label.endswith("LOC"):
            masked_words.append("[LOC]")
        elif ner_label.endswith("PER"):
            masked_words.append("[PERSON]")
        elif ner_label.endswith("ORG"):
            masked_words.append("[ORG]")
        else:
            masked_words.append(token)

    return " ".join(masked_words)


# def mask_entities(question: str):
#     """
#     Mask câu hỏi: giữ động từ/so sánh, thay thế thực thể bằng token riêng,
#     còn lại mask thành [MASK]
#     """
#     key_pos = {"V", "A", "M", "Cc"}  # Động từ, tính từ/so sánh, số lượng, liên từ
#     masked_tokens = []
#     annotated = model.annotate_text(question)
#     # annotated là dict: {0: [ {wordForm, posTag, nerLabel, head, depLabel}, ... ]}

#     for sid, sentence in annotated.items():
#         for token in sentence:
#             word = token["wordForm"].replace("_", " ")

#             pos = token["posTag"]
#             ner_label = token["nerLabel"]

#             # Ưu tiên mask theo thực thể
#             if ner_label.endswith("LOC"):
#                 masked_tokens.append("[LOC]")
#             elif ner_label.endswith("PER"):
#                 masked_tokens.append("[PERSON]")
#             elif ner_label.endswith("ORG"):
#                 masked_tokens.append("[ORG]")
#             else:
#                 # Mask theo POS
#                 if pos in key_pos:
#                     masked_tokens.append(word.lower())  # Giữ động từ/so sánh
#                 else:
#                     masked_tokens.append("[MASK]")      # Mask các token khác

#     return " ".join(masked_tokens)


# Ví dụ sử dụng hàm
if __name__ == "__main__":
    question = "liệt kê tên và họ của các cầu thủ đã chơi cho các đội có sân vận động ở California, và đã từng thắng World Series ít nhất một lần"
    #     # question= "Liệt kê tên của những cá nhân quê ở Đà Nẵng và liên quan đến các vụ tội phạm có hơn 1 người bị thương"
    database_name = "baseball_1"
    # question = mask_entities(question)
    # print(question)
    desc_exemplars,_ = retrieve_from_collections(db_folder=r"D:\vitext2sql_vi\vitext2sql\pre\db\baseball_1",db_path="",db_des=True, question=question, database_name=database_name,log_callback=None)
        
    # Save prompt context
    save_prompt_context(
        db_des=True,
        db_folder=r"D:\vitext2sql_vi\vitext2sql\pre\db\baseball_1",
        db_path="",
        results=desc_exemplars, 
        id=1234, 
        db_name=database_name,log_callback=None
    )

    # print("Retrieved elements:")
    # for result in results:
    #     print(f"ID: {result['id']}, Type: {result['type']}, Content: {result['content']}, Distance: {result['distance']}, Metadata: {result['metadata']}")
