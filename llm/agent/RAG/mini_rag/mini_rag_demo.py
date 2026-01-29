# mini_rag_demo.py
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import textwrap

# ========== (0) 数据结构：Chunk ==========
@dataclass
class Chunk:
    id: str
    text: str
    meta: Dict[str, Any]

# ========== (1)(2) 向量库：build + query ==========
class MiniVectorStore:
    def __init__(self):
        self.chunks: List[Chunk] = []
        self.vect = TfidfVectorizer(ngram_range=(1, 2))
        self.X = None

    def upsert(self, docs: List[Dict[str, Any]]):
        # docs: [{"id": "...", "text": "...", ...meta}]
        self.chunks = [Chunk(d["id"], d["text"], {k:v for k,v in d.items() if k not in ("id","text")}) for d in docs]
        self.X = self.vect.fit_transform([c.text for c in self.chunks])

    def query(self, q: str, top_k: int = 3) -> List[Tuple[Chunk, float]]:
        qv = self.vect.transform([q])
        sims = cosine_similarity(qv, self.X).ravel()
        idxs = sims.argsort()[::-1][:top_k]
        return [(self.chunks[i], float(sims[i])) for i in idxs]

# ========== (4) LLM：这里用模板模拟 ==========
class TemplateLLM:
    def generate(self, prompt: str) -> str:
        # 真实项目里这里是 Gemini / OpenAI 等
        return "（模板LLM输出）\n" + prompt

# ========== (3) Pipeline：拼 Prompt + 生成 ==========
class MiniRAGPipeline:
    def __init__(self, store: MiniVectorStore, llm=None):
        self.store = store
        self.llm = llm or TemplateLLM()

    def build_prompt(self, question: str, passages: List[Chunk]) -> str:
        ctx = "\n\n".join([f"[{c.id}] {c.text}" for c in passages])
        prompt = f"Context:\n{ctx}\n\nQuestion:\n{question}\n\nAnswer:"
        return textwrap.shorten(prompt, width=1200, placeholder="\n...\n")

    def answer(self, question: str, top_k: int = 3):
        hits = self.store.query(question, top_k=top_k)
        passages = [c for c, _ in hits]
        prompt = self.build_prompt(question, passages)
        return self.llm.generate(prompt), hits

if __name__ == "__main__":
    # ====== 准备一个小 KB（模拟你们的知识库 chunks）======
    kb_docs = [
        {"id":"kb1", "text":"UR3 报错 C204A 通常与关节通信或急停相关，建议检查安全回路与电源。", "tag":"ur3"},
        {"id":"kb2", "text":"如果相机初始化失败，先检查 USB 带宽与 RealSense pipeline 是否正确启动。", "tag":"camera"},
        {"id":"kb3", "text":"恢复策略建议：先复位急停->重新上电->重连控制器->执行回零动作。", "tag":"strategy"},
    ]

    store = MiniVectorStore()
    store.upsert(kb_docs)

    rag = MiniRAGPipeline(store)
    ans, hits = rag.answer("UR3急停后怎么恢复？", top_k=2)

    print("=== 检索结果 ===")
    for c, s in hits:
        print(c.id, "score=", round(s, 3), "text=", c.text)

    print("\n=== 生成输出 ===")
    print(ans)
