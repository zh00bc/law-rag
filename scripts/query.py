"""
查询脚本：语义检索 ChromaDB 中的税法条文，输出 JSON。
服务器上复用 OpenClaw 已有的 OPENAI_API_KEY。
"""

import argparse
import json
import os
import sys
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

# 尝试加载 .env（本地测试用；服务器上 OpenClaw 已设好环境变量）
for env_path in [
    Path(__file__).resolve().parent.parent / '.env',
    Path(__file__).resolve().parent.parent.parent / '.env',
]:
    if env_path.exists():
        load_dotenv(env_path)
        break


def get_collection(db_path: str) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=db_path)
    return client.get_collection(name='chinese_tax_law')


def embed_query(query: str, client: OpenAI, model: str = 'text-embedding-3-small') -> list[float]:
    resp = client.embeddings.create(input=[query], model=model)
    return resp.data[0].embedding


def search(collection: chromadb.Collection, query_embedding: list[float],
           top_k: int = 5, law_filter: str | None = None) -> list[dict]:
    where = None
    if law_filter:
        where = {'law_short_name': {'$eq': law_filter}}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where,
        include=['documents', 'metadatas', 'distances'],
    )

    output = []
    for i in range(len(results['ids'][0])):
        distance = results['distances'][0][i]
        score = round(1 - distance, 4)  # cosine distance → similarity
        meta = results['metadatas'][0][i]
        # 从 text 中去掉 【...】 前缀，返回原始条文
        text = results['documents'][0][i]
        if text.startswith('【'):
            newline_pos = text.find('\n')
            if newline_pos != -1:
                text = text[newline_pos + 1:]

        output.append({
            'text': text,
            'law_name': meta.get('law_name', ''),
            'law_short_name': meta.get('law_short_name', ''),
            'chapter': meta.get('chapter', ''),
            'article_number': meta.get('article_number', ''),
            'effective_date': meta.get('effective_date', ''),
            'score': score,
        })

    return output


def main():
    parser = argparse.ArgumentParser(description='查询中国税法条文')
    parser.add_argument('--query', required=True, help='查询问题')
    parser.add_argument('--top-k', type=int, default=5, help='返回结果数量')
    parser.add_argument('--law', default=None, help='按法律简称过滤（如 增值税法）')
    parser.add_argument('--db-path', default=None, help='ChromaDB 路径')
    args = parser.parse_args()

    # 默认 db_path 在 skill 目录下
    db_path = args.db_path or str(Path(__file__).resolve().parent.parent / 'chroma_db')

    if not Path(db_path).exists():
        print(json.dumps({'error': f'ChromaDB not found at {db_path}. Run setup.sh first.'}))
        sys.exit(1)

    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print(json.dumps({'error': 'OPENAI_API_KEY not set'}))
        sys.exit(1)

    base_url = os.environ.get('OPENAI_BASE_URL', 'https://openrouter.ai/api/v1')
    client = OpenAI(api_key=api_key, base_url=base_url)
    collection = get_collection(db_path)

    query_emb = embed_query(args.query, client)
    results = search(collection, query_emb, top_k=args.top_k, law_filter=args.law)

    print(json.dumps({
        'query': args.query,
        'results_count': len(results),
        'results': results,
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
