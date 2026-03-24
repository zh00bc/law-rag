"""
本地预处理脚本：解析税法 markdown → 按条分块 → 生成上下文描述 → embedding → 导出 chunks.json

改进：
- Contextual Embeddings（Anthropic 方法）：用 LLM 为每个 chunk 生成上下文描述
- 上下文描述拼接在条文前面后再做 embedding，提升检索精度
"""

import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# ── 常量 ──

ARTICLE_RE = re.compile(r'^(第[一二三四五六七八九十百零]+条)\s', re.MULTILINE)
CHAPTER_RE = re.compile(r'^(第[一二三四五六七八九十]+章)\s+(.+)', re.MULTILINE)
FULLWIDTH_SPACE_RE = re.compile(r'[\u3000\s]+')
APPENDIX_RE = re.compile(r'^(附表[一二三四五六七八九十]*：|附：)\s*$', re.MULTILINE)
TOC_MARKER = '目\u3000\u3000录'  # 目　　录

CN_DIGITS = {
    '零': 0, '一': 1, '二': 2, '三': 3, '四': 4,
    '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
}


def cn_num_to_int(s: str) -> int:
    """'七十二' → 72, '十' → 10, '三' → 3, '一百' → 100"""
    s = s.strip()
    if not s:
        return 0
    # 百位
    if '百' in s:
        parts = s.split('百')
        h = CN_DIGITS.get(parts[0], 1)
        rest = parts[1] if len(parts) > 1 else ''
        return h * 100 + (cn_num_to_int(rest) if rest else 0)
    # 十位
    if '十' in s:
        parts = s.split('十')
        tens = CN_DIGITS.get(parts[0], 1) if parts[0] else 1
        ones = CN_DIGITS.get(parts[1], 0) if len(parts) > 1 and parts[1] else 0
        return tens * 10 + ones
    if s.startswith('零'):
        return cn_num_to_int(s[1:])
    return CN_DIGITS.get(s, 0)


def parse_article_number(art: str) -> int:
    """'第七十二条' → 72"""
    inner = art.replace('第', '').replace('条', '')
    return cn_num_to_int(inner)


# ── 文件名解析 ──

def parse_filename(filepath: Path) -> dict:
    stem = filepath.stem  # 中华人民共和国增值税法_20241225
    parts = stem.rsplit('_', 1)
    law_name = parts[0]
    date_str = parts[1] if len(parts) > 1 else ''
    effective_date = f'{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}' if len(date_str) == 8 else ''
    law_short_name = law_name.replace('中华人民共和国', '')
    return {
        'law_name': law_name,
        'law_short_name': law_short_name,
        'effective_date': effective_date,
        'source_file': filepath.name,
    }


# ── 正文提取 ──

def extract_body(text: str) -> str:
    """跳过标题和目录，返回正文（从第一个正文章节或条文开始）"""
    lines = text.split('\n')

    # 找目录标记
    toc_idx = None
    for i, line in enumerate(lines):
        if TOC_MARKER in line:
            toc_idx = i
            break

    if toc_idx is not None:
        # 有目录：找第二个「第一章」或第一个「第一条」（取先出现者）
        first_chapter_count = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if re.match(r'^第一章\s', stripped):
                first_chapter_count += 1
                if first_chapter_count == 2:
                    return '\n'.join(lines[i:])
            # 如果找到了第一条但还没找到第二个第一章，也可以开始
            if first_chapter_count == 1 and i > toc_idx and re.match(r'^第一条\s', stripped):
                return '\n'.join(lines[i:])
        # fallback: 从目录后第一个第X章或第X条开始
        for i in range(toc_idx + 1, len(lines)):
            if ARTICLE_RE.match(lines[i].strip()) or CHAPTER_RE.match(lines[i].strip()):
                return '\n'.join(lines[i:])
    else:
        # 无目录：从第一个第X章或第X条开始
        for i, line in enumerate(lines):
            stripped = line.strip()
            if CHAPTER_RE.match(stripped) or ARTICLE_RE.match(stripped):
                return '\n'.join(lines[i:])

    return text  # fallback


# ── 分块 ──

def chunk_document(filepath: Path) -> list[dict]:
    meta_base = parse_filename(filepath)
    text = filepath.read_text(encoding='utf-8')
    body = extract_body(text)

    # 分离附表
    appendix_match = APPENDIX_RE.search(body)
    if appendix_match:
        main_body = body[:appendix_match.start()]
        appendix_text = body[appendix_match.start():]
    else:
        # 也检查 "附：" 后面直接跟内容的情况（如关税法）
        inline_appendix = re.search(r'^(附：.+)', body, re.MULTILINE)
        if inline_appendix:
            main_body = body[:inline_appendix.start()]
            appendix_text = body[inline_appendix.start():]
        else:
            main_body = body
            appendix_text = None

    # 构建 chapter 位置映射
    chapters = list(CHAPTER_RE.finditer(main_body))
    chapter_positions = [(m.start(), m.group(1) + ' ' + FULLWIDTH_SPACE_RE.sub('', m.group(2).strip())) for m in chapters]

    def get_chapter(pos: int) -> str:
        ch = ''
        for cp, cn in chapter_positions:
            if cp <= pos:
                ch = cn
        return ch

    # 按条切分
    articles = list(ARTICLE_RE.finditer(main_body))
    chunks = []

    for i, match in enumerate(articles):
        start = match.start()
        end = articles[i + 1].start() if i + 1 < len(articles) else len(main_body)

        article_text = main_body[start:end].strip()
        article_number = match.group(1)
        article_index = parse_article_number(article_number)
        chapter = get_chapter(start)

        # 前缀加法律名和章节
        prefix = f'【{meta_base["law_short_name"]}'
        if chapter:
            prefix += f' {chapter}'
        prefix += '】\n'

        chunk_id = f'{meta_base["law_short_name"]}_{meta_base["effective_date"]}_{article_index:03d}'

        chunks.append({
            'id': chunk_id,
            'text': prefix + article_text,
            'metadata': {
                **meta_base,
                'chapter': chapter,
                'article_number': article_number,
                'article_index': article_index,
                'chunk_type': 'article',
            },
        })

    # 附表
    if appendix_text and appendix_text.strip():
        # 按 "附表X：" 或 "附：" 分割
        appendix_parts = re.split(r'^(附表[一二三四五六七八九十]*：|附：)', appendix_text, flags=re.MULTILINE)
        # appendix_parts: ['', '附表一：', content, '附表二：', content, ...]
        idx = 1
        app_num = 0
        while idx < len(appendix_parts):
            app_label = appendix_parts[idx].strip().rstrip('：:')
            app_content = appendix_parts[idx + 1] if idx + 1 < len(appendix_parts) else ''
            app_num += 1

            full_text = f'【{meta_base["law_short_name"]} {app_label}】\n{app_label}\n{app_content.strip()}'
            chunk_id = f'{meta_base["law_short_name"]}_{meta_base["effective_date"]}_app{app_num:02d}'

            chunks.append({
                'id': chunk_id,
                'text': full_text,
                'metadata': {
                    **meta_base,
                    'chapter': '',
                    'article_number': app_label,
                    'article_index': 9000 + app_num,  # 附表排在最后
                    'chunk_type': 'appendix',
                },
            })
            idx += 2

    return chunks


# ── Contextual Embeddings ──

CONTEXT_PROMPT = """你是一个中国税法专家。请为以下法律条文生成一段简短的上下文描述（50-100字），用于辅助语义检索。

要求：
1. 说明该条文出自哪部法律、哪个章节
2. 用通俗语言概括该条文的核心内容
3. 列出 3-5 个关键检索词
4. 不要复述条文原文

法律名称：{law_name}
章节：{chapter}
条号：{article_number}

条文内容：
{article_text}

请直接输出上下文描述，不要加任何前缀或格式标记。"""


def generate_contexts(chunks: list[dict], client: OpenAI, chat_model: str = 'openai/gpt-4.1-mini') -> list[str]:
    """为每个 chunk 生成上下文描述"""
    contexts = []
    for i, chunk in enumerate(chunks):
        meta = chunk['metadata']
        # 从 text 中提取原始条文（去掉 【...】 前缀）
        raw_text = chunk['text']
        if raw_text.startswith('【'):
            nl = raw_text.find('\n')
            if nl != -1:
                raw_text = raw_text[nl + 1:]

        prompt = CONTEXT_PROMPT.format(
            law_name=meta.get('law_name', ''),
            chapter=meta.get('chapter', ''),
            article_number=meta.get('article_number', ''),
            article_text=raw_text[:1500],  # 截断超长附表
        )

        ctx = ''
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=chat_model,
                    messages=[{'role': 'user', 'content': prompt}],
                    max_tokens=200,
                    temperature=0.3,
                )
                ctx = resp.choices[0].message.content.strip()
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    print(f'  Warning: context generation failed for {chunk["id"]}: {e}')

        contexts.append(ctx)

        if (i + 1) % 20 == 0 or i + 1 == len(chunks):
            print(f'  Generated context {i + 1}/{len(chunks)}')

    failed = sum(1 for c in contexts if not c)
    if failed:
        print(f'  Warning: {failed}/{len(chunks)} chunks have empty context')

    return contexts


# ── Embedding ──

def generate_embeddings(chunks: list[dict], client: OpenAI, model: str = 'text-embedding-3-small', batch_size: int = 50) -> list[list[float]]:
    texts = [c['text'] for c in chunks]
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        print(f'  Embedding batch {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1} ({len(batch)} chunks)...')
        resp = client.embeddings.create(input=batch, model=model)
        all_embeddings.extend([d.embedding for d in resp.data])

    return all_embeddings


# ── Main ──

def main():
    # 加载 .env（先找项目根目录，再找上级目录）
    for env_path in [
        Path(__file__).resolve().parent.parent / '.env',
        Path(__file__).resolve().parent.parent.parent / '.env',
    ]:
        if env_path.exists():
            load_dotenv(env_path)
            break

    docs_dir = Path(__file__).resolve().parent.parent / 'docs'
    output_path = Path(__file__).resolve().parent.parent / 'data' / 'chunks.json'

    if not docs_dir.exists():
        print(f'Error: docs directory not found at {docs_dir}')
        sys.exit(1)

    md_files = sorted(docs_dir.glob('*.md'))
    if not md_files:
        print(f'Error: no .md files found in {docs_dir}')
        sys.exit(1)

    # 解析和分块
    all_chunks = []
    print('Parsing and chunking documents...\n')
    for f in md_files:
        chunks = chunk_document(f)
        articles = [c for c in chunks if c['metadata']['chunk_type'] == 'article']
        appendices = [c for c in chunks if c['metadata']['chunk_type'] == 'appendix']
        print(f'  {f.name}: {len(articles)} articles, {len(appendices)} appendix chunks')
        all_chunks.extend(chunks)

    print(f'\nTotal: {len(all_chunks)} chunks\n')

    # 生成 embedding
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print('Error: OPENAI_API_KEY not set. Add it to .env or export it.')
        sys.exit(1)

    base_url = os.environ.get('OPENAI_BASE_URL', 'https://openrouter.ai/api/v1')
    client = OpenAI(api_key=api_key, base_url=base_url)
    print(f'Using API: {base_url}')

    # 生成上下文描述（Contextual Embeddings）
    chat_model = os.environ.get('CHAT_MODEL', 'openai/gpt-4.1-mini')
    print(f'\nGenerating contextual descriptions using {chat_model}...')
    contexts = generate_contexts(all_chunks, client, chat_model=chat_model)

    # 将上下文拼接到 chunk text 前面
    for chunk, ctx in zip(all_chunks, contexts):
        if ctx:
            # 保留原始条文在 raw_text 字段，方便 BM25 和展示
            chunk['raw_text'] = chunk['text']
            # 新的 text = 上下文描述 + 原始条文（用于 embedding）
            chunk['text'] = ctx + '\n\n' + chunk['text']
            chunk['context'] = ctx
        else:
            chunk['raw_text'] = chunk['text']
            chunk['context'] = ''

    print('\nGenerating embeddings...')
    embeddings = generate_embeddings(all_chunks, client)

    # 写入 JSON
    for chunk, emb in zip(all_chunks, embeddings):
        chunk['embedding'] = emb

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f'\nExported {len(all_chunks)} chunks to {output_path} ({size_mb:.1f} MB)')


if __name__ == '__main__':
    main()
