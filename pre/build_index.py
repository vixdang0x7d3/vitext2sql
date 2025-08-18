import os
import sqlite3
import json
from typing import List, Tuple
import re
import unicodedata
import numpy as np
# from sentence_transformers import SentenceTransformer
import chromadb

# ======================== CẤU HÌNH ========================
# DB_URI = "sqlite:///your_database.db"   # demo SQLite file path ở dưới
# SQLITE_PATH = r"D:\vitext2sql_vi\vitext2sql\pre\db\perpetrator\perpetrator.sqlite"        # đường dẫn tệp SQLite thực
TABLE_INCLUDE = None  # ví dụ: ["users", "orders"] hoặc None = lấy tất cả
TEXT_COL_MAX_CHARS = 256  # cắt bớt chuỗi quá dài (giảm nhiễu)
EMBED_BATCH = 256

# LSH (SimHash) cấu hình
NUM_BITS = 64           # tổng bit simhash
NUM_BANDS = 8           # số band
BITS_PER_BAND = NUM_BITS // NUM_BANDS  # = 8
RNG_SEED = 42

# Lưu trữ
PERSIST_DIR = "./lsh_semantic"
CHROMA_DIR = os.path.join(PERSIST_DIR, "chroma")
META_DIR = os.path.join(PERSIST_DIR, "meta")
BUCKETS_DB = os.path.join(PERSIST_DIR, "lsh_buckets.sqlite")
HYPERPLANES_NPY = os.path.join(META_DIR, "hyperplanes.npy")
os.makedirs(PERSIST_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)

MODEL_NAME = "BAAI/bge-m3"

# ======================== TIỆN ÍCH DB ========================
def list_tables_sqlite(conn) -> List[str]:
    cur = conn.cursor()
    rs = cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    tbls = [r[0] for r in rs if not r[0].startswith("sqlite_")]
    if TABLE_INCLUDE:
        tbls = [t for t in tbls if t in TABLE_INCLUDE]
    return tbls

def list_columns_sqlite(conn, table: str) -> List[str]:
    cur = conn.cursor()
    rs = cur.execute(f'PRAGMA table_info("{table}")').fetchall()
    cols = [r[1] for r in rs]
    return cols

def distinct_values_sqlite(conn, table: str, col: str) -> List[str]:
    cur = conn.cursor()
    try:
        rs = cur.execute(f'SELECT DISTINCT "{col}" FROM "{table}" WHERE "{col}" IS NOT NULL').fetchall()
        print(f"[DEBUG] Column: {col} | First 5 rows: {rs[:5]}")
    except Exception:
        return []
    vals = []
    for (v,) in rs:
        s = str(v).strip()
        if not s:
            continue
        if len(s) > TEXT_COL_MAX_CHARS:
            s = s[:TEXT_COL_MAX_CHARS]
        vals.append(s)
    # khử trùng lặp “thực sự”
    seen, out = set(), []
    for v in vals:
        if v not in seen:
            seen.add(v); out.append(v)
    return out

# ======================== LSH (SimHash) ========================
def ensure_hyperplanes(dim: int) -> np.ndarray:
    """
    Sinh (hoặc nạp) ma trận hyperplanes: shape (NUM_BITS, dim).
    Giữ nguyên giữa các lần build để hash ổn định.
    """
    if os.path.exists(HYPERPLANES_NPY):
        hp = np.load(HYPERPLANES_NPY)
        if hp.shape == (NUM_BITS, dim):
            return hp
    rng = np.random.default_rng(RNG_SEED)
    hp = rng.standard_normal((NUM_BITS, dim)).astype(np.float32)
    np.save(HYPERPLANES_NPY, hp)
    return hp

def simhash_bits(emb: np.ndarray, hyperplanes: np.ndarray) -> np.ndarray:
    """
    emb: (dim,) đã normalize
    hyperplanes: (NUM_BITS, dim)
    return: mảng bit (NUM_BITS,) kiểu uint8 {0,1}
    """
    proj = hyperplanes @ emb  # (NUM_BITS,)
    bits = (proj > 0).astype(np.uint8)
    return bits

def band_signatures(bits: np.ndarray) -> List[str]:
    """Cắt bits thành NUM_BANDS band, mỗi band BITS_PER_BAND bit → hex string để lưu gọn."""
    sigs = []
    for b in range(NUM_BANDS):
        start = b * BITS_PER_BAND
        end = start + BITS_PER_BAND
        band_bits = bits[start:end]
        # chuyển sang int rồi sang hex, cố định chiều dài
        val = 0
        for bit in band_bits:
            val = (val << 1) | int(bit)
        sigs.append(f"{val:0{BITS_PER_BAND//4}x}")  # ví dụ 8 bit -> 2 hex
    return sigs

# ======================== BUCKET STORE (SQLite) ========================
def init_bucket_store(db_name=""):
    BUCKETS_DB = os.path.join(PERSIST_DIR, f"{db_name}_lsh_buckets.sqlite")
    conn = sqlite3.connect(BUCKETS_DB)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS buckets (
        band_index INTEGER,
        signature TEXT,
        item_id TEXT
    )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_buckets ON buckets( band_index, signature)")
    conn.commit()
    conn.close()

def insert_bucket_entries(entries: List[Tuple[int,str,str]],db_name = ""):
    """
    entries: list of ( band_index, signature, item_id)
    """
    BUCKETS_DB = os.path.join(PERSIST_DIR, f"{db_name}_lsh_buckets.sqlite")
    conn = sqlite3.connect(BUCKETS_DB)
    cur = conn.cursor()
    cur.executemany(
        "INSERT INTO buckets(band_index,signature,item_id) VALUES (?,?,?)",
        entries
    )
    conn.commit()
    conn.close()
def normalize_text(s):
    
    s = s.lower()
    s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    s = re.sub(r'[^\w\s]', ' ', s)  # bỏ dấu câu
    return ' '.join(s.split())
# ======================== BUILD QUY TRÌNH ========================
# def main():
#     # 1) Mở DB nguồn (SQLite demo)
#     src = sqlite3.connect(SQLITE_PATH)

#     # 2) Chroma collection “semantic_values”
#     chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
#     coll = chroma_client.get_or_create_collection(name="semantic_values")

#     # reset (tuỳ chọn)
#     try:
#         coll.delete(where={})
#     except Exception:
#         pass

#     # 3) Load model, chuẩn bị hyperplanes
#     model = SentenceTransformer(MODEL_NAME)
#     # encode thử để biết dim
#     probe = model.encode(["probe"], normalize_embeddings=True)
#     dim = probe.shape[1]
#     hp = ensure_hyperplanes(dim)

#     init_bucket_store()

#     # 4) Duyệt tất cả bảng/cột
#     item_ids = []
#     docs = []
#     metas = []
#     embs = []

#     for table in list_tables_sqlite(src):
#         for col in list_columns_sqlite(src, table):
#             values = distinct_values_sqlite(src, table, col)
#             if not values:
#                 continue
#             print(f"[DEBUG] Table: {table}, Column: {col}, Values count: {len(values)}")        
#             # Embed theo batch
#             for i in range(0, len(values), EMBED_BATCH):
#                 chunk = values[i:i+EMBED_BATCH]
#                 vecs = model.encode(chunk, normalize_embeddings=True)
#                 print("[DEBUG] First embedding vector:", vecs[0])
#                 print("[DEBUG] Embedding shape:", vecs.shape)
#                 # add vào Chroma + chuẩn bị bucket LSH
#                 entries = []
#                 for j, (val, emb) in enumerate(zip(chunk, vecs)):
#                     item_id = f"{table}|{col}|{i+j}"
#                     print(f"[DEBUG] Processing: {table}|{col}|{i+j} -> {item_id}")
#                     item_ids.append(item_id)
#                     docs.append(val)
#                     print(val)
#                     metas.append({"table": table, "column": col})
#                     embs.append(emb.tolist())
#                     print(emb)
#                     # print(emb.tolist())

#                     bits = simhash_bits(emb, hp)
#                     bits_str = ''.join(str(b) for b in bits)
#                     print("[DEBUG] SimHash bits:", bits_str)
#                     sigs = band_signatures(bits)
#                     for b_idx, sig in enumerate(sigs):
#                         print(f"[DEBUG] Band {b_idx}: {sig}")
#                     for b_idx, sig in enumerate(sigs):
                        
#                         entries.append((table, col, b_idx, sig, item_id))

#                 # flush các entries band vào SQLite
#                 if entries:
#                     insert_bucket_entries(entries)
#                     print(f"[DEBUG] Inserting {len(entries)} bucket entries into SQLite")

#     # flush vào Chroma
#     if item_ids:
#         print(f"[DEBUG] Adding {len(item_ids)} items to Chroma")
#         print(f"[DEBUG] First doc: {docs[0]}")
#         print(f"[DEBUG] First embedding shape: {np.array(embs).shape}")
#         coll.add(ids=item_ids, documents=docs, metadatas=metas, embeddings=embs)

#     print(f"✅ Build xong. Chroma tại: {CHROMA_DIR}")
#     print(f"✅ Buckets LSH tại: {BUCKETS_DB}")
#     print(f"✅ Hyperplanes: {HYPERPLANES_NPY}")
def main(db_path="",db_name="",model=None):
    import sqlite3, hashlib


    src = sqlite3.connect(db_path)

    # Chroma collection
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    coll = chroma_client.create_collection(name=db_name)

    # Reset collection
    try:
        coll.delete(where={})
    except Exception:
        pass

    # model = SentenceTransformer(MODEL_NAME)
    dim = len(model.encode(["probe"])[0])
    hp = ensure_hyperplanes(dim)
    init_bucket_store(db_name=db_name)

    # ---- 1) Lấy tất cả unique values trong DB ----
    all_values = set()
    for table in list_tables_sqlite(src):
        for col in list_columns_sqlite(src, table):
            vals = distinct_values_sqlite(src, table, col)
            all_values.update(v for v in vals if v and v.strip())

    all_values = list(all_values)
    print(f"✅ Tổng unique values: {len(all_values)}")

    # ---- 2) Embed + LSH + Chroma ----
    item_ids, docs, metas, embs = [], [], [], []

    for i in range(0, len(all_values), EMBED_BATCH):
        chunk = all_values[i:i+EMBED_BATCH]
        chunk_norm_text = [normalize_text(i) for i in chunk]
        vecs = model.encode(chunk_norm_text)

        entries = []
        for val, emb in zip(chunk, vecs):
            # ID có thể là hash để đảm bảo unique
            item_id = hashlib.md5(val.encode("utf-8")).hexdigest()
            item_ids.append(item_id)
            docs.append(val)
            # metas.append({})  
            embs.append(emb)

            bits = simhash_bits(emb, hp)
            sigs = band_signatures(bits)
            for b_idx, sig in enumerate(sigs):
                entries.append((b_idx, sig, item_id))  

        if entries:
            insert_bucket_entries(entries,db_name=db_name)

    # ---- 3) Lưu vào Chroma ----
    if item_ids:
        # print(embs)
        coll.add(ids=item_ids, documents=docs, embeddings=embs)

    print(f"✅ Build xong. Chroma: {CHROMA_DIR}")
    print(f"✅ Buckets LSH: {BUCKETS_DB}")
    print(f"✅ Hyperplanes: {HYPERPLANES_NPY}")
if __name__ == "__main__":
    main()