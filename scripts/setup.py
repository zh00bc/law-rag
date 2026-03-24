"""
服务器安装脚本：从 chunks.json 导入预计算数据到 ChromaDB。
"""

import argparse
import json
import sys
from pathlib import Path

import chromadb


def main():
    parser = argparse.ArgumentParser(description='导入预计算数据到 ChromaDB')
    parser.add_argument('--data', default=None, help='chunks.json 路径')
    parser.add_argument('--db-path', default=None, help='ChromaDB 存储路径')
    args = parser.parse_args()

    skill_dir = Path(__file__).resolve().parent.parent
    data_path = Path(args.data) if args.data else skill_dir / 'data' / 'chunks.json'
    db_path = args.db_path or str(skill_dir / 'chroma_db')

    if not data_path.exists():
        print(f'Error: {data_path} not found')
        sys.exit(1)

    print(f'Loading chunks from {data_path}...')
    with open(data_path, encoding='utf-8') as f:
        chunks = json.load(f)

    print(f'Loaded {len(chunks)} chunks')

    # 创建 ChromaDB
    client = chromadb.PersistentClient(path=db_path)

    # 删除旧 collection（如果存在）并重建
    try:
        client.delete_collection('chinese_tax_law')
        print('Deleted existing collection')
    except Exception:
        pass

    collection = client.create_collection(
        name='chinese_tax_law',
        metadata={'hnsw:space': 'cosine'},
    )

    # 批量 upsert
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        collection.add(
            ids=[c['id'] for c in batch],
            documents=[c['text'] for c in batch],
            embeddings=[c['embedding'] for c in batch],
            metadatas=[c['metadata'] for c in batch],
        )
        print(f'  Imported {min(i + batch_size, len(chunks))}/{len(chunks)} chunks')

    print(f'\nDone! {collection.count()} chunks in ChromaDB at {db_path}')


if __name__ == '__main__':
    main()
