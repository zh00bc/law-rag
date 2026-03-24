"""
查询脚本：混合检索（BM25 + 语义）ChromaDB 中的税法条文，输出 JSON。

改进：
- Hybrid Search：BM25 关键词匹配 + ChromaDB 语义检索
- Reciprocal Rank Fusion (RRF) 合并排序
- 返回原始条文（去掉上下文前缀）
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi

# 尝试加载 .env（本地测试用；服务器上 OpenClaw 已设好环境变量）
for env_path in [
    Path(__file__).resolve().parent.parent / '.env',
    Path(__file__).resolve().parent.parent.parent / '.env',
]:
    if env_path.exists():
        load_dotenv(env_path)
        break


# ── 简易中文分词 ──

def tokenize_chinese(text: str) -> list[str]:
    """简单的中文分词：按标点切分 + 2-gram"""
    # 去掉标点和空白
    text = re.sub(r'[^\u4e00-\u9fff\w]', ' ', text)
    words = text.split()
    tokens = []
    for w in words:
        tokens.append(w)
        # 对中文部分生成 bigram
        for i in range(len(w) - 1):
            tokens.append(w[i:i+2])
    return tokens


# ── BM25 索引 ──

class BM25Index:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        # 使用 context（上下文描述）+ raw_text 做 BM25
        # context 包含关键检索词，弥补长条文中关键词被稀释的问题
        texts = []
        for c in chunks:
            ctx = c.get('context', '')
            raw = c.get('raw_text', c['text'])
            texts.append(ctx + '\n' + raw if ctx else raw)
        tokenized = [tokenize_chinese(t) for t in texts]
        self.bm25 = BM25Okapi(tokenized)
        self.ids = [c['id'] for c in chunks]

    def search(self, query: str, top_k: int = 20, law_filter: str | None = None) -> list[tuple[str, float]]:
        tokens = tokenize_chinese(query)
        scores = self.bm25.get_scores(tokens)
        results = list(zip(self.ids, scores))
        if law_filter:
            id_set = {c['id'] for c in self.chunks if c['metadata'].get('law_short_name') == law_filter}
            results = [(cid, s) for cid, s in results if cid in id_set]
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


# ── Reciprocal Rank Fusion ──

def reciprocal_rank_fusion(ranked_lists: list[list[tuple[str, float]]], weights: list[float] | None = None, k: int = 60) -> list[tuple[str, float]]:
    """加权合并多个排序列表，返回 (id, rrf_score) 列表"""
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    scores = {}
    for ranked, weight in zip(ranked_lists, weights):
        for rank, (doc_id, _) in enumerate(ranked, 1):
            scores[doc_id] = scores.get(doc_id, 0) + weight / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ── ChromaDB ──

def get_collection(db_path: str) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=db_path)
    return client.get_collection(name='chinese_tax_law')


def embed_query(query: str, client: OpenAI, model: str = 'text-embedding-3-small') -> list[float]:
    resp = client.embeddings.create(input=[query], model=model)
    return resp.data[0].embedding


def semantic_search(collection: chromadb.Collection, query_embedding: list[float],
                    top_k: int = 20, law_filter: str | None = None) -> list[tuple[str, float]]:
    where = None
    if law_filter:
        where = {'law_short_name': {'$eq': law_filter}}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where,
        include=['distances'],
    )

    output = []
    for i in range(len(results['ids'][0])):
        score = 1 - results['distances'][0][i]
        output.append((results['ids'][0][i], score))
    return output


def extract_display_text(text: str) -> str:
    """从 chunk text 中提取原始条文用于展示（去掉 context 和 【...】 前缀）"""
    # 去掉 contextual prefix（在 【 之前的部分）
    bracket_pos = text.find('【')
    if bracket_pos > 0:
        text = text[bracket_pos:]
    # 去掉 【...】 行
    if text.startswith('【'):
        nl = text.find('\n')
        if nl != -1:
            text = text[nl + 1:]
    return text.strip()


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description='查询中国税法条文')
    parser.add_argument('--query', required=True, help='查询问题')
    parser.add_argument('--top-k', type=int, default=5, help='返回结果数量')
    parser.add_argument('--law', default=None, help='按法律简称过滤（如 增值税法）')
    parser.add_argument('--db-path', default=None, help='ChromaDB 路径')
    args = parser.parse_args()

    skill_dir = Path(__file__).resolve().parent.parent
    db_path = args.db_path or str(skill_dir / 'chroma_db')
    data_path = skill_dir / 'data' / 'chunks.json'

    if not Path(db_path).exists():
        print(json.dumps({'error': f'ChromaDB not found at {db_path}. Run setup.sh first.'}))
        sys.exit(1)

    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print(json.dumps({'error': 'OPENAI_API_KEY not set'}))
        sys.exit(1)

    base_url = os.environ.get('OPENAI_BASE_URL', 'https://openrouter.ai/api/v1')
    client = OpenAI(api_key=api_key, base_url=base_url)

    # 加载 chunks 用于 BM25 和结果展示
    with open(data_path, encoding='utf-8') as f:
        chunks = json.load(f)
    chunk_map = {c['id']: c for c in chunks}

    # 构建 BM25 索引
    bm25_index = BM25Index(chunks)

    # 语义检索
    collection = get_collection(db_path)
    query_emb = embed_query(args.query, client)
    semantic_results = semantic_search(collection, query_emb, top_k=20, law_filter=args.law)

    # BM25 检索
    bm25_results = bm25_index.search(args.query, top_k=20, law_filter=args.law)

    # 加权 RRF 合并（语义权重 3x，BM25 权重 1x — 语义为主，BM25 补充关键词匹配）
    fused = reciprocal_rank_fusion([semantic_results, bm25_results], weights=[3.0, 1.0])

    # 取 top-k 并构建输出
    output = []
    for doc_id, rrf_score in fused[:args.top_k]:
        chunk = chunk_map.get(doc_id)
        if not chunk:
            continue
        meta = chunk['metadata']
        article_text = extract_display_text(chunk.get('raw_text', chunk['text']))
        law_name = meta.get('law_name', '')
        article_number = meta.get('article_number', '')
        # 预格式化引用，LLM 可直接使用
        citation = f'《{law_name}》{article_number}'
        output.append({
            'citation': citation,
            'text': article_text,
            'chapter': meta.get('chapter', ''),
            'effective_date': meta.get('effective_date', ''),
            'score': round(rrf_score, 4),
        })

    print(json.dumps({
        'query': args.query,
        'results_count': len(output),
        'results': output,
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
