# 📞 来电助手 — Hybrid RAG 智能来电分类系统

> 基于混合 RAG（检索增强生成）+ Whisper ASR 的智能来电分类与处理系统。  
> 核心场景：用户上课期间，自动识别来电类型，并给出代接/拦截/记录等处理建议。

---

## 🎬 演示视频

> 📥 演示视频已移至 [GitHub Releases](https://github.com/QM-newer/YanXi-KCN/releases)，请前往下载 `demo.mp4`

---

## ✨ 核心功能

- 🎙️ **语音识别（ASR）**：基于 OpenAI Whisper（small 模型），支持麦克风实时录音和音频文件识别
- 🧠 **智能分类**：RAG 向量检索 + 关键词匹配 + LLM 增强，三级分类策略覆盖 **16 种来电类型**
- 📋 **处理规则**：针对每种来电类型预设处理动作（代接/拦截/记录/优先处理/询问）
- 🔍 **混合检索**：ChromaDB 向量检索 + NetworkX 知识图谱 + Louvain 社区检测 + RRF 融合
- 🤖 **LLM 问答**：基于通义千问 qwen-turbo 的上下文感知答案生成
- 🛡️ **无意义检测**：自动过滤纯数字、重复字符、计数序列等无效语音输入
- 📊 **离线索引**：预构建 1550 条通话记录的向量库 + 2836 节点知识图谱 + 15 个社区检测

---

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆仓库
git clone <your-repo-url>
cd call_hybrid_rag

# 创建虚拟环境
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # macOS/Linux

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置 API Key

复制 `.env.example` 为 `.env` 并填写你的通义千问 API Key：

```env
QWEN_API_KEY=sk-your-api-key-here
```

> 仅使用 RAG 分类（不调用 LLM）时可不配置。

### 3. 构建离线索引（首次使用）

```bash
# 准备通话数据
python scripts/prepare_call_data.py

# 构建向量索引 + 知识图谱 + 社区检测
python scripts/build_indices.py
```

构建完成后将在 `indices/` 目录下生成：
- `vector_store_v6/` — 通话记录向量库（1550 条）
- `graph.pkl` — 知识图谱（2836 节点，3231 边）
- `communities.json` — 社区检测结果（15 个社区）
- `summary_store/` — 社区摘要向量库

---

## 🎤 语音分类（推荐入口）

### 麦克风模式

```bash
cd voice
python voice_call.py mic
```

按键操作：按 **空格键** 开始录音 → 说话 → 再按 **空格键** 停止 → 自动分类

### 音频文件模式

```bash
cd voice
python voice_call.py file test.mp3
```

### 交互模式

```bash
cd voice
python voice_call.py
```

输入文本即可分类，输入 `q` 退出。

### 生成测试音频

```bash
cd voice
python generate_test_audio.py
# 将生成 test_audio.mp3 并自动播放
```

---

## 📖 来电分类类型

系统支持 **16 种来电类型**，每种类型有预设处理动作：

| 来电类型 | 处理动作 | 处理规则 |
|---------|---------|---------|
| 外卖配送 | 代接 | 告知配送员放东门传达室，提取配送信息发短信通知 |
| 快递取件 | 代接 | 告知配送员放东门传达室或快递柜，提取配送信息发短信通知 |
| 打车到达 | 代接 | 告知稍等，记录短信通知 |
| 推销电话 | 拦截 | 直接挂断，推送推销来电提醒短信 |
| 诈骗电话 | 拦截 | 直接拦截挂断，推送风险提醒短信 |
| 诈骗风险 | 拦截 | 直接拦截挂断，推送风险提醒短信 |
| 游戏周年庆 | 拦截 | 直接挂断，标记为营销来电 |
| 熟人问候 | 记录 | 询问是否有急事，不紧急晚点联系，紧急记录通知 |
| 同事协作 | 记录 | 询问是否紧急，不紧急留言，紧急记录通知 |
| 客户来电 | 记录 | 询问是否有紧急事项，记录留言通知 |
| 银行电话 | 记录 | 询问是否有紧急事项，记录留言通知 |
| 领导来电 | 优先处理 | 询问是否有急事，记录留言后发短信通知回电 |
| 面试通知 | 优先处理 | 询问面试时间和联系方式，记录通知 |
| 家人电话 | 询问 | 询问是否紧急，不紧急晚点联系 |
| 无意义 | 询问 | 告知没有听清，请对方重复一遍 |
| 其他 | 询问 | 告知正在上课，询问来电事由并记录 |

---

## 📁 项目结构

```
call_hybrid_rag/
├── voice/                         # 🎙️ 语音模块
│   ├── asr.py                     #   Whisper ASR（文件 + 麦克风录音）
│   ├── voice_call.py              #   语音分类主入口（mic/file/interactive）
│   ├── generate_test_audio.py     #   Edge-TTS 测试音频生成
│   └── __init__.py
├── src/                           # 🧠 核心引擎
│   ├── agents/
│   │   └── call_classifier.py     #   电话分类 Agent（16 类 + 无意义检测）
│   ├── hybrid_rag.py              #   Hybrid RAG 主类（路由→检索→融合→生成）
│   ├── pipeline.py                #   来电助手主流程
│   ├── factory.py                 #   依赖注入工厂
│   ├── data/                      #   数据处理（加载器 + 清洗器）
│   ├── indexing/                  #   离线索引构建（向量 + 图谱 + 社区）
│   ├── retrieval/                 #   检索模块（路由 + 向量 + 图谱 + 融合 + 重排）
│   ├── generation/                #   LLM 答案生成
│   └── utils/                     #   工具（配置 + 日志 + LLM 客户端）
├── scripts/                       # 📜 命令行脚本
│   ├── build_indices.py           #   索引构建
│   ├── prepare_call_data.py       #   数据预处理
│   ├── classifier.py              #   分类器交互测试
│   └── query.py                   #   RAG 交互查询
├── configs/
│   └── config.yaml                #   总配置文件
├── indices/                       #   预构建的索引文件
├── tests/                         #   单元测试
├── requirements.txt
└── README.md
```

---

## 🧪 技术栈

| 类别 | 技术 | 用途 |
|------|------|------|
| ASR | OpenAI Whisper (small) | 语音转文字，中文优化 |
| TTS | Edge-TTS | 测试音频生成 |
| 录音 | sounddevice / pyaudio | 麦克风实时录音 |
| 向量数据库 | ChromaDB | 通话记录向量存储 + 社区摘要存储 |
| Embedding | BGE-large-zh-v1.5 | 中文文本向量化（1024 维） |
| 知识图谱 | NetworkX + python-louvain | 实体关系图 + Louvain 社区检测 |
| LLM API | 通义千问 qwen-turbo | 分类增强 + 答案生成 |
| 检索融合 | RRF (Reciprocal Rank Fusion) | 向量检索 + 图谱检索结果融合 |

---

## 🔧 可用命令

```bash
# === 语音分类 ===
python voice/voice_call.py mic              # 麦克风按键录音分类
python voice/voice_call.py mic --auto 10    # 自动录音 10 秒分类
python voice/voice_call.py file test.mp3    # 音频文件分类
python voice/voice_call.py                  # 交互式文本分类

# === 独立 ASR 测试 ===
python -m voice.asr mic                     # 麦克风语音转文字
python -m voice.asr file test.mp3           # 音频文件转文字

# === RAG 查询 ===
python scripts/query.py                     # 交互式 RAG 问答

# === 分类器测试 ===
python scripts/classifier.py                # 测试 16 类分类

# === 索引构建 ===
python scripts/build_indices.py             # 构建全部索引
```

---

## ⚠️ 注意事项

- **API Key**：LLM 增强功能需要配置通义千问 API Key，纯 RAG 分类可不配置
- **模型下载**：首次运行 ASR 时，Whisper 会自动下载 `small` 模型（~1GB），请确保网络畅通
- **pyaudio**：录音库优先使用 sounddevice，pyaudio 作为 Windows 上的备选方案
- **索引文件**：`indices/` 目录已预构建完成，如需重建请运行 `python scripts/build_indices.py`

---

## 📄 License

MIT License
