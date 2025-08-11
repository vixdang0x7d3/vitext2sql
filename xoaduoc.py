import re
import json

with open(r"D:\vitext2sql_vi\vitext2sql\pre\train.json", "r", encoding="utf-8") as f:
    data = json.load(f)

def remove_punctuation(text):
    text = text.lower()
    # Xóa dấu phẩy, chấm, chấm phẩy, chấm than, hỏi
    text = re.sub(r"[.,;!?]", " ", text)
    # Nhưng giữ lại dấu . trong tên người hoặc số liệu
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Tạo danh sách documents
documents = []
with open(r"D:\vitext2sql_vi\vitext2sql\pre\tables.json", "r", encoding="utf-8") as f:
    schemas = json.load(f)



# Từ khóa SQL không cần quote
sql_keywords = {
    "select", "from", "where", "join", "on", "as", "and", "or",
    "=", ">", "<", ">=", "<=", "<>", "!=", "*"
}
def quote_token(token: str) -> str:
    """Quote tên bảng/cột nếu cần"""
    stripped = token.strip()
    lower = stripped.lower()

    # Không quote nếu là từ khóa SQL hoặc số hoặc đã có quote
    if (lower in sql_keywords
        or re.match(r'^[0-9]+$', stripped)
        or stripped.startswith('"')
        or stripped.startswith("'")):
        return token

    # Nếu có dấu .
    if "." in stripped:
        parts = stripped.split(".")
        quoted_parts = []
        for part in parts:
            if part in table_set or part in column_set:
                quoted_parts.append(f'"{part}"')
            else:
                quoted_parts.append(part)
        return ".".join(quoted_parts)

    # Không có dấu . thì kiểm tra trực tiếp
    if stripped in table_set or stripped in column_set:
        return f'"{stripped}"'

    return token

for item in data:
    question = remove_punctuation(item.get("question", "")).strip()
    # query = item.get("query", "").strip()
    db_id = item.get("db_id", "").strip()

    # Lấy schema của academic
    academic_schemas = next(s for s in schemas if s["db_id"] == item["db_id"])

    # Tập hợp tên bảng và tên cột
    table_set = set(academic_schemas["table_names_original"])
    column_set = set(col[1] for col in academic_schemas["column_names_original"])
    quoted_tokens = [quote_token(tok) for tok in item["query_toks"]]
    token_based_query = " ".join(quoted_tokens).strip()

    if question and token_based_query:
        doc = {
            "text": question,
            "metadata": {
                "sql_text": token_based_query,
                "db_id": db_id,
            }
        }
        documents.append(doc)

# # Đọc schema
# with open(r"D:\vitext2sql_vi\vitext2sql\pre\tables.json", "r", encoding="utf-8") as f:
#     schemas = json.load(f)

# # Lấy schema của academic
# academic_schemas = next(s for s in schemas if s["db_id"] == "academic")

# # Tập hợp tên bảng và tên cột
# table_set = set(academic_schemas["table_names_original"])
# column_set = set(col[1] for col in academic_schemas["column_names_original"])

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

# # Test
# sample_data = {
#     "db_id": "academic",
#     "query_toks": [
#         "select", "t1.tên", "from", "bài báo", "as", "t4",
#         "join", "tạp chí", "as", "t2", "on", "t4.id tạp chí", "=",
#         "t2.id tạp chí", "join", "viết", "as", "t3", "on",
#         "t3.id bài báo", "=", "t4.id bài báo", "join", "tác giả",
#         "as", "t1", "on", "t3.id tác giả", "=", "t1.id tác giả",
#         "where", "t2.tên", "=", "\"PVLDB\""
#     ]
# }

# quoted_tokens = [quote_token(tok) for tok in sample_data["query_toks"]]
# token_based_query = " ".join(quoted_tokens)

# print("Quoted Query:")
# print(token_based_query)
