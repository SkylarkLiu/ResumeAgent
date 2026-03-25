# RAG 系统架构与优化策略

## 一、RAG 概述

RAG（Retrieval-Augmented Generation，检索增强生成）通过**外部知识检索**增强 LLM 的回答能力，解决幻觉、知识过期、私有数据等问题。

### 核心流程

```
用户问题 → Query 处理 → 向量检索 → 上下文拼装 → LLM 生成 → 回答
              ↓              ↓                          ↓
          改写/拆分     Top-K 召回                引用来源标注
```

## 二、Query 预处理

### 2.1 Query 改写（Query Rewriting）

将用户自然语言问题转换为更适合检索的查询。

```python
REWRITE_PROMPT = """
你是一个搜索查询优化专家。将用户的自然语言问题改写为更适合语义检索的查询。
要求：简洁、去歧义、提取关键词、保留核心语义。

用户问题：{question}
改写后的查询：
"""
```

### 2.2 Query 拆分（Query Decomposition）

复杂问题拆分为多个子查询，分别检索后合并。

```python
def decompose_query(question: str) -> list[str]:
    """将复杂问题拆分为 2-3 个子查询"""
    # 示例："Python 和 Go 在微服务场景下哪个更适合？"
    # → ["Python 微服务性能特点", "Go 微服务性能特点", "Python vs Go 微服务对比"]
    pass
```

### 2.3 HyDE（Hypothetical Document Embedding）

让 LLM 先生成一个假设性答案，再用这个答案做检索（答案比问题更接近文档语义）。

```python
def hyde_search(question: str, embed_fn, retriever):
    # 1. LLM 生成假设答案
    hypothesis = llm.generate(f"请回答：{question}")
    # 2. 用假设答案做向量检索
    results = retriever.search(embed_fn(hypothesis), top_k=5)
    return results
```

## 三、文本分块策略

### 3.1 分块方法对比

| 方法 | 原理 | 适用场景 | 缺点 |
|------|------|----------|------|
| **固定长度** | 按字符/Token 数切分 | 通用 | 可能切断语义 |
| **递归字符** | 按分隔符层级切分（段落→句子→字符） | 推荐，通用性强 | 需要调参 |
| **语义分块** | 按语义相似度聚类切分 | 长文档、学术论文 | 计算成本高 |
| **文档结构** | 按 Markdown 标题/HTML 标签切分 | 结构化文档 | 依赖文档格式 |
| **句子窗口** | 每句作为独立块，上下文作为元数据 | 精确检索 | 块数量多 |

### 3.2 RecursiveCharacterTextSplitter

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # 每块最大字符数
    chunk_overlap=50,     # 重叠字符数（保持上下文连贯）
    separators=["\n\n", "\n", "。", ".", " ", ""]  # 切分优先级
)
```

### 3.3 分块参数调优

| 参数 | 影响 | 建议值 |
|------|------|--------|
| `chunk_size` | 太小 → 语义不完整；太大 → 检索噪声多 | 300-800 |
| `chunk_overlap` | 太小 → 上下文断裂；太大 → 冗余 | chunk_size 的 10-20% |
| `separators` | 影响切分粒度 | 优先段落 → 句子 → 字符 |

## 四、向量检索

### 4.1 Embedding 模型选型

| 模型 | 维度 | 特点 |
|------|------|------|
| OpenAI text-embedding-3-small | 1536 | 性价比高，英文强 |
| OpenAI text-embedding-3-large | 3072 | 精度高，成本较高 |
| 智谱 embedding-3 | 2048 | 中文优化，OpenAI 兼容 API |
| BGE-large-zh-v1.5 | 1024 | 开源，中文优秀，本地部署 |

### 4.2 相似度计算

| 方法 | 公式 | 特点 |
|------|------|------|
| **余弦相似度** | cos(A,B) = A·B / (\|A\|\|B\|) | 最常用，不受向量长度影响 |
| **内积 (IP)** | A·B | FAISS IndexFlatIP + L2 归一化 = 余弦相似度 |
| **L2 距离** | \|A-B\|² | 距离越小越相似，FAISS 默认 |

### 4.3 FAISS 索引类型

| 索引 | 说明 | 适用规模 |
|------|------|----------|
| `IndexFlatIP` | 暴力搜索，精确但慢 | < 100 万条 |
| `IndexIVFFlat` | 倒排索引 + 暴力 | 100 万 - 1000 万 |
| `IndexHNSW` | 层次导航小图图 | 1000 万+ |
| `IndexPQ` | 乘积量化，压缩存储 | 超大规模，牺牲精度 |

### 4.4 元数据管理

向量库只存 embedding，元数据（文件名、页码、内容）用侧文件存储。

```python
# vector_store 结构
data/faiss_index/
├── index.faiss      # FAISS 索引
└── metadata.json    # 元数据侧文件
```

## 五、检索优化

### 5.1 Top-K 与阈值过滤

```python
def retrieve(query_embedding, top_k=5, score_threshold=0.5):
    results = vector_store.search(query_embedding, top_k * 2)
    # 过滤低分结果
    return [r for r in results if r["score"] >= score_threshold][:top_k]
```

### 5.2 重排序（Reranking）

初检索 Top-20 → 精排模型 → 最终 Top-5。

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("BAAI/bge-reranker-base")

def rerank(query, candidates, top_k=5):
    pairs = [(query, c["content"]) for c in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
    return ranked[:top_k]
```

### 5.3 MMR（最大边际相关性）

平衡**相关性**与**多样性**，避免检索结果高度重复。

```python
from langchain_community.vectorstores import FAISS

results = vector_store.max_marginal_relevance_search(
    query, k=5, fetch_k=20, lambda_mult=0.5
)
```

### 5.4 多路召回（Hybrid Search）

向量检索 + 关键词检索（BM25）→ 结果融合。

```python
def hybrid_search(query, top_k=5):
    # 向量检索
    vector_results = vector_search(query, top_k * 2)
    # 关键词检索
    bm25_results = bm25_search(query, top_k * 2)
    # RRF（Reciprocal Rank Fusion）融合
    return rrf_merge(vector_results, bm25_results, top_k)
```

## 六、上下文拼装

### 6.1 标准模板

```python
CONTEXT_TEMPLATE = """基于以下参考内容回答问题。如果参考内容中没有相关信息，请根据你的知识回答，并说明"该信息未在知识库中找到"。

{context}

用户问题：{question}
"""
```

### 6.2 上下文格式

```
【来源1 - 知识库】文件名.pdf 第3页
这是第一段检索到的内容...

【来源2 - 知识库】文件名.pdf 第5页
这是第二段检索到的内容...

【来源3 - 网络搜索】https://example.com
这是网络搜索到的内容...
```

### 6.3 上下文长度管理

```python
def truncate_context(sources, max_chars=4000):
    """根据 Token 预算截断上下文"""
    total = 0
    selected = []
    for source in sources:
        if total + len(source["content"]) > max_chars:
            source["content"] = source["content"][:max_chars - total]
            selected.append(source)
            break
        total += len(source["content"])
        selected.append(source)
    return selected
```

## 七、生成优化

### 7.1 System Prompt 设计原则

| 原则 | 说明 |
|------|------|
| 明确边界 | 告诉模型"只基于参考内容回答" |
| 引用标注 | 要求模型标注信息来源 |
| 承认不确定 | 允许模型说"不知道"而非编造 |
| 格式控制 | 指定输出格式（Markdown/表格/JSON） |

### 7.2 流式输出

```python
async def stream_answer(question, context):
    async for chunk in llm.astream(prompt):
        yield chunk.content  # SSE 推送到前端
```

## 八、评估指标

### 8.1 检索质量

| 指标 | 说明 |
|------|------|
| **Recall@K** | 相关文档被召回的比例 |
| **MRR** | 第一个相关文档排名的倒数 |
| **NDCG** | 考虑排名位置的归一化折损增益 |

### 8.2 生成质量

| 指标 | 说明 |
|------|------|
| **Faithfulness** | 回答是否忠于检索内容（不编造） |
| **Answer Relevancy** | 回答与问题的相关性 |
| **Context Relevancy** | 检索内容与问题的相关性 |

### 8.3 RAGAS 评估框架

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_relevancy

result = evaluate(
    dataset=test_dataset,
    metrics=[faithfulness, answer_relevancy, context_relevancy]
)
```
