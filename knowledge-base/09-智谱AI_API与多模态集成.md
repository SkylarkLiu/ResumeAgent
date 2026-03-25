# 智谱 AI API 与多模态集成

## 一、智谱 AI 简介

智谱 AI（Zhipu AI）提供与 OpenAI 兼容的 API 接口，支持文本生成、Embedding、视觉理解等多种能力。通过 `zai-sdk`（新版包名）可以直接调用。

### 核心模型

| 模型 | 用途 | 特点 |
|------|------|------|
| `glm-4-flash` | 文本生成 | 快速、低成本，适合 Agent 路由/生成 |
| `glm-4-plus` | 文本生成（增强） | 更高准确度，适合复杂推理 |
| `glm-4v-flash` | 视觉理解 | 支持图片输入，OCR + 理解 |
| `embedding-3` | 文本嵌入 | 2048 维，中文优化 |
| `embedding-2` | 文本嵌入（轻量） | 1024 维，速度更快 |

## 二、API 基础

### 2.1 安装与初始化

```bash
pip install zai-sdk
```

```python
from zai import ZhipuAiClient

client = ZhipuAiClient(api_key="your_api_key")
```

### 2.2 文本生成（Chat Completion）

```python
response = client.chat.completions.create(
    model="glm-4-flash",
    messages=[
        {"role": "system", "content": "你是一个有用的助手"},
        {"role": "user", "content": "你好"}
    ],
    temperature=0.7,
    max_tokens=2048
)

answer = response.choices[0].message.content
```

### 2.3 消息格式（OpenAI 兼容）

```python
messages = [
    {"role": "system", "content": "系统指令"},
    {"role": "user", "content": "用户消息"},
    {"role": "assistant", "content": "助手回复"},
    {"role": "user", "content": "后续提问"}
]
```

### 2.4 Embedding

```python
response = client.embeddings.create(
    model="embedding-3",
    input=["文本1", "文本2", "文本3"]
)

vectors = [item.embedding for item in response.data]
# 每个向量维度: 2048
```

**注意事项**：
- 单次最多 64 条
- 单条文本最大 8192 Token
- 超出需分批处理

## 三、视觉理解（GLM-4V-Flash）

### 3.1 图片输入方式

GLM-4V-Flash 支持三种图片输入格式：

| 方式 | 格式 | 适用场景 |
|------|------|----------|
| **URL** | `"image_url": {"url": "https://..."}` | 公网图片 |
| **Base64** | `"image_url": {"url": "data:image/png;base64,..."}` | 本地图片、前端上传 |
| **文件路径** | 先读取文件转 base64 | 服务端文件处理 |

### 3.2 Base64 调用示例

```python
import base64

def vision_chat(prompt: str, image_base64: str) -> str:
    """图片 base64 视觉理解"""
    client = ZhipuAiClient(api_key=settings.zhipuai_api_key)
    
    response = client.chat.completions.create(
        model="glm-4v-flash",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"}
                    }
                ]
            }
        ],
        temperature=0.5,
        max_tokens=2048
    )
    
    return response.choices[0].message.content
```

### 3.3 本地文件调用

```python
def vision_from_file(prompt: str, image_path: str) -> str:
    """读取本地图片文件进行视觉理解"""
    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")
    
    return vision_chat(prompt, image_base64)
```

### 3.4 结构化图片理解

```python
IMAGE_UNDERSTANDING_PROMPT = """请分析这张图片，按以下结构输出：

## 摘要
简要描述图片的主要内容（50字以内）

## 关键元素
- 元素1：描述
- 元素2：描述

## 文字识别（OCR）
提取图片中所有可见的文字内容

## 表格/图表信息
如果图片包含表格或图表，请提取其结构化数据

## 关键词
[关键词1, 关键词2, 关键词3]

## 文件来源推断
推断这个文件可能的来源或类型
"""
```

## 四、PDF 双模处理

### 4.1 文本型 PDF

```python
import fitz  # PyMuPDF

def extract_text(pdf_path: str) -> str:
    """提取 PDF 文本"""
    doc = fitz.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc):
        text = page.get_text()
        pages.append(f"--- 第 {page_num + 1} 页 ---\n{text}")
    return "\n\n".join(pages)
```

### 4.2 扫描件 PDF（图片型）

```python
def render_page_as_image(doc, page_num: int, dpi: int = 200) -> bytes:
    """将 PDF 页面渲染为 PNG"""
    page = doc[page_num]
    pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
    return pix.tobytes("png")

def process_scanned_pdf(pdf_path: str, vision_fn) -> str:
    """扫描件 PDF 逐页视觉理解"""
    doc = fitz.open(pdf_path)
    results = []
    
    for page_num in range(len(doc)):
        img_bytes = render_page_as_image(doc, page_num)
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
        
        page_text = vision_chat(
            f"请提取第 {page_num + 1} 页的全部文字内容",
            img_base64
        )
        results.append(f"--- 第 {page_num + 1} 页 ---\n{page_text}")
    
    return "\n\n".join(results)
```

### 4.3 自动检测 PDF 类型

```python
def is_scanned_pdf(doc, text_threshold: int = 50) -> bool:
    """判断 PDF 是否为扫描件"""
    total_chars = 0
    sample_pages = min(5, len(doc))
    
    for i in range(sample_pages):
        text = doc[i].get_text().strip()
        total_chars += len(text)
    
    avg_chars = total_chars / sample_pages
    return avg_chars < text_threshold
```

### 4.4 统一处理流程

```python
def process_pdf(pdf_path: str) -> str:
    """PDF 智能处理：自动判断类型"""
    doc = fitz.open(pdf_path)
    
    if is_scanned_pdf(doc):
        # 扫描件 → 逐页渲染 → 视觉理解
        return process_scanned_pdf(pdf_path, vision_chat)
    else:
        # 文本型 → 直接提取
        return extract_text(pdf_path)
```

## 五、API 调用优化

### 5.1 错误处理

```python
import time
from zai.errors import APIError

def call_with_retry(func, max_retries=3, base_delay=1.0):
    """指数退避重试"""
    for attempt in range(max_retries):
        try:
            return func()
        except APIError as e:
            if e.status_code == 429:  # Rate Limit
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
            elif e.status_code >= 500:  # Server Error
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
            else:
                raise
    raise Exception(f"API 调用失败，已重试 {max_retries} 次")
```

### 5.2 并发控制

```python
import asyncio
import semver

async def batch_embed(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """并发批量 Embedding"""
    semaphore = asyncio.Semaphore(5)  # 最多 5 个并发
    
    async def embed_batch(batch):
        async with semaphore:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, lambda: embed_texts(batch)
            )
    
    batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
    results = await asyncio.gather(*[embed_batch(b) for b in batches])
    
    return [vec for batch_result in results for vec in batch_result]
```

### 5.3 Token 计算

```python
def estimate_tokens(text: str) -> int:
    """估算 Token 数（中文约 1.5 字符/token）"""
    return int(len(text) * 1.5)

def check_context_limit(messages: list, model: str = "glm-4-flash") -> list:
    """裁剪消息历史，控制在上下文窗口内"""
    MAX_TOKENS = {
        "glm-4-flash": 128000,
        "glm-4v-flash": 8000,  # 视觉模型窗口较小
    }
    
    limit = MAX_TOKENS.get(model, 128000)
    total = sum(estimate_tokens(m.get("content", "")) for m in messages)
    
    while total > limit * 0.8 and len(messages) > 1:
        messages.pop(0)  # 移除最早的消息
        total = sum(estimate_tokens(m.get("content", "")) for m in messages)
    
    return messages
```

## 六、多模态在 Agent 中的应用

### 6.1 图片即问答

用户发送图片 → GLM-4V-Flash 直接理解 → 返回描述/回答

```
用户上传图片 → base64 编码 → vision_chat(question, base64) → 回答
```

### 6.2 图片入知识库

用户上传图片 → 视觉理解提取文本描述 → 文本分块 → Embedding → 入 FAISS

```
图片 → understand_image() → 结构化文本 → 分块 → 向量化 → 入库
```

### 6.3 简历多模态提取

```
简历（PDF/图片/文本）→ 统一转为文本 → LLM 结构化提取 → JSON
  ├── .txt/.md → 直接读取
  ├── .png/.jpg → GLM-4V-Flash OCR
  └── .pdf → 文本型直接提取 / 扫描件渲染图片走视觉
```

## 七、成本与限额

### 7.1 模型定价参考

| 模型 | 输入价格 | 输出价格 | 适用场景 |
|------|----------|----------|----------|
| glm-4-flash | ¥0.1/百万Token | ¥0.1/百万Token | Agent 路由、通用生成 |
| glm-4-plus | ¥0.5/百万Token | ¥0.5/百万Token | 复杂推理、报告生成 |
| glm-4v-flash | ¥0.1/百万Token | ¥0.1/百万Token | 图片理解、OCR |
| embedding-3 | ¥0.5/百万Token | — | 向量化 |

### 7.2 配额管理

```python
class TokenBudget:
    def __init__(self, daily_limit: int = 1_000_000):
        self.daily_limit = daily_limit
        self.used_today = 0
    
    def check(self, estimated_tokens: int) -> bool:
        return self.used_today + estimated_tokens <= self.daily_limit
    
    def record(self, tokens: int):
        self.used_today += tokens
```
