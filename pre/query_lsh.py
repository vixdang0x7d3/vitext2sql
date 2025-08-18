import os
import sqlite3
from typing import List, Dict, Any
import re
import unicodedata
import numpy as np
# from sentence_transformers import SentenceTransformer
import chromadb

# ====== CẤU HÌNH ======
PERSIST_DIR = "./lsh_semantic"
CHROMA_DIR = os.path.join(PERSIST_DIR, "chroma")
BUCKETS_DB = os.path.join(PERSIST_DIR, "lsh_buckets.sqlite")
HYPERPLANES_NPY = os.path.join(PERSIST_DIR, "meta", "hyperplanes.npy")

MODEL_NAME = "BAAI/bge-m3"
NUM_BITS = 64
NUM_BANDS = 8
BITS_PER_BAND = NUM_BITS // NUM_BANDS

# ====== LSH Utils ======
def simhash_bits(emb: np.ndarray, hyperplanes: np.ndarray) -> np.ndarray:
    proj = hyperplanes @ emb
    return (proj > 0).astype(np.uint8)

def band_signatures(bits: np.ndarray) -> List[str]:
    sigs = []
    for b in range(NUM_BANDS):
        start = b * BITS_PER_BAND
        end = start + BITS_PER_BAND
        band_bits = bits[start:end]
        val = 0
        for bit in band_bits:
            val = (val << 1) | int(bit)
        sigs.append(f"{val:0{BITS_PER_BAND//4}x}")
    return sigs
def normalize_text(s):
    
    s = s.lower()
    s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    s = re.sub(r'[^\w\s]', ' ', s)  # bỏ dấu câu
    return ' '.join(s.split())

# ====== QUERY PIPELINE ======
class LSHChromaNormalizer:
    def __init__(self,db_path="",db_name="",model=None):
        self.model = model
        self.hp = np.load(HYPERPLANES_NPY).astype(np.float32)
        self.client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.coll = self.client.get_collection(name=db_name)
        print("[DEBUG] Collection count:", self.coll.count())
        print("[DEBUG] First 5 IDs in collection:", self.coll.get(ids=None, limit=5)["ids"])
        print(os.path.join(PERSIST_DIR, f"{db_name}_lsh_buckets.sqlite"))
        self.buckets = sqlite3.connect( os.path.join(PERSIST_DIR, f"{db_name}_lsh_buckets.sqlite"))

    def _candidate_ids_from_bands(self,band_sigs: List[str], band_limit_per_sig=200) -> List[str]:
        cur = self.buckets.cursor()
        ids = []
        for b_idx, sig in enumerate(band_sigs):
            rs = cur.execute(
                """SELECT item_id FROM buckets 
                   WHERE band_index=? AND signature=?
                   LIMIT ?""",
                ( b_idx, sig, band_limit_per_sig)
            ).fetchall()
            ids.extend([r[0] for r in rs])
        # khử trùng lặp, giữ thứ tự
        seen, out = set(), []
        for _id in ids:
            if _id not in seen:
                seen.add(_id); out.append(_id)
        return out

    def normalize(self,  user_value: str, top_k: int = 5) -> Dict[str, Any]:
        user_value=normalize_text(user_value)
        # 1) encode & hash
        q = np.array(self.encode([user_value])[0], dtype=np.float32)
        bits = simhash_bits(q, self.hp)
        sigs = band_signatures(bits)

        # 2) lấy danh sách ứng viên qua bucket LSH
        cand_ids = self._candidate_ids_from_bands(sigs)
        print(f"[DEBUG] cand_ids ({len(cand_ids)}):", cand_ids[:10])  # <-- thêm debug
        if not cand_ids:
            return {"rep": user_value, "candidates": [], "note": "no bucket hit; fallback to input"}
        print(cand_ids[0])
        # 3) lấy embeddings ứng viên từ Chroma
        got = self.coll.get(
            ids=cand_ids,
            include=["embeddings", "documents", "metadatas"]
        )
        docs = got["documents"]
        embs = np.array(got["embeddings"], dtype=np.float32)
        print(got.keys())
        print(f"[DEBUG] docs ({len(docs)}):", docs)              # <-- thêm debug
        print("[DEBUG] embs raw:", got["embeddings"])            # <-- thêm debug
        print("[DEBUG] embs shape:", getattr(embs, "shape", None))
        
        # 4) rerank theo cosine trong tập nhỏ
        # q = q.astype(np.float32)
        print("[DEBUG] q shape:", getattr(q, "shape", None))
        sims = (embs @ q).tolist()
        items = list(zip(cand_ids, docs, sims))
        items.sort(key=lambda x: x[2], reverse=True)
        top = items[:top_k]

        # rep = max((t[1] for t in top), key=lambda s: (len(s), s.lower()))

        return {
            "rep": [d for i, d, s in top],
            "candidates": [{"id": i, "value": d, "cosine": float(s)} for i, d, s in top]
        }

# ====== DEMO ======
if __name__ == "__main__":
    norm = LSHChromaNormalizer()
    # ví dụ:
    # for q in ["Việt Nam", "Viet Nam", "USA", "Nippon"]:
    #     ans = norm.normalize(table="tội phạm", column="địa điểm", user_value=q, top_k=5)
    #     print("\nQuery:", q)
    #     print("Rep  :", ans["rep"])
    #     print("Top  :", ans["candidates"][:3])

    ans = norm.normalize( user_value="1", top_k=10)
    # print("\nQuery:", q)
    print("Rep  :", ans["rep"])
    print("Top  :", ans["candidates"][:3])


