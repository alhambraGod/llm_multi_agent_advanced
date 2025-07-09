# 健身助手Agent

基于LlamaIndex、Langchain和Langgraph打造的智能健身助手，采用GraphRAG知识图谱技术，通过火山方舟Deepseek大模型为用户提供专业健身指导。

## 核心功能

### 智能对话系统
- **专业健身咨询**: 回答训练计划、饮食建议、动作指导等健身相关问题
- **个性化建议**: 基于用户历史对话记录提供定制化建议
- **多轮对话**: 支持上下文理解，实现自然流畅的多轮交流
- **知识溯源**: 显示回答来源，提高可信度和透明度

### RAG增强检索
- **GraphRAG知识图谱**: 结合知识图谱的检索增强生成技术，提升回答准确性
- **高质量向量搜索**: 使用BAAI/bge-m3中文embedding模型进行语义检索
- **检索结果重排序**: 通过重排序算法优化检索结果的相关性
- **多源知识融合**: 整合健身专业知识与用户个人训练数据

### 用户体验
- **现代化UI界面**: 基于Next.js构建的响应式前端设计
- **实时对话反馈**: 即时响应用户提问，提供加载状态展示
- **会话管理**: 支持清空对话、查看历史记录等功能
- **多终端适配**: 自适应不同屏幕尺寸的设备

### 安全与隐私
- **多租户权限隔离**: 确保用户数据安全，个人信息仅限本人访问
- **JWT认证机制**: 保障API访问安全性
- **数据加密存储**: 敏感信息加密保存

## 技术架构

### 后端技术栈
- **框架**: FastAPI
- **Agent引擎**: LlamaIndex + Langchain + Langgraph
- **大语言模型**: 火山方舟Deepseek
- **向量数据库**: Milvus
- **Embedding模型**: BAAI/bge-m3
- **容器化**: Docker + Docker Compose

### 前端技术栈
- **框架**: Next.js + React
- **样式**: Tailwind CSS
- **状态管理**: React Context API
- **HTTP客户端**: Axios
- **UI组件**: React Icons, React Markdown
- **认证管理**: JS-Cookie

## 详细架构设计

### 后端组件
- **Agent工作流管理器**: 协调知识检索、记忆检索和响应生成的流程
- **GraphRAG模块**: 结合知识图谱的检索增强生成
  - **健身知识导入**: 将肌肉群、训练动作、健身计划、营养建议等专业知识结构化处理并转换为向量存储
  - **用户训练数据导入**: 将用户资料、训练日志、身体测量数据和营养记录等个人数据向量化存储
  - **Embedding技术**: 使用BAAI/bge-m3模型将文本转换为语义向量，支持高效相似度搜索
  - **多源检索融合**: 同时从健身专业知识库和用户个人数据中检索相关信息并按相关度排序
  - **元数据过滤**: 支持基于用户ID等属性的精准数据访问控制
- **对话记忆系统**: 存储和检索用户历史对话，支持上下文理解
- **重排序模块**: 优化检索结果相关性
- **权限控制系统**: 基于JWT的用户认证和授权

### 前端模块
- **认证上下文**: 管理用户登录状态与权限
- **聊天界面**: 实现实时对话功能
- **侧边栏导航**: 提供应用内各功能的快捷访问
- **健身数据展示**: 可视化用户健身记录和进度

## 目录结构
```
fitness_llm_agent_opt/
├── backend/                  # 后端服务
│   ├── agent/                # Agent核心实现
│   │   ├── core.py           # Agent工作流定义
│   │   ├── llm.py            # 大模型接口适配
│   │   └── workflows/        # 对话流程定义
│   ├── auth/                 # 认证授权
│   │   └── auth.py           # JWT认证实现
│   ├── database/             # 数据库操作
│   │   └── milvus_client.py  # Milvus客户端
│   ├── embeddings/           # 嵌入模型
│   │   └── embeddings.py     # 向量嵌入接口
│   ├── evaluation/           # 评估工具
│   │   └── evaluator.py      # RAG和Agent评估
│   ├── knowledge/            # 知识库管理
│   │   ├── graphrag.py       # GraphRAG实现
│   │   └── data/             # 健身知识数据
│   ├── memory/               # 会话记忆
│   │   └── memory.py         # 记忆存储与检索
│   ├── rerank/               # 结果重排序
│   │   └── rerank.py         # 重排序算法
│   ├── app.py                # 主应用入口
│   └── Dockerfile            # 后端容器配置
├── frontend/                 # 前端应用
│   ├── src/                  # 源代码
│   │   ├── components/       # UI组件
│   │   ├── context/          # 上下文管理
│   │   ├── pages/            # 页面定义
│   │   └── styles/           # 样式文件
│   ├── public/               # 静态资源
│   ├── tailwind.config.js    # Tailwind配置
│   └── Dockerfile            # 前端容器配置
├── docker-compose.yml        # 服务编排配置
└── requirements.txt          # Python依赖
```

## 安装部署

### 环境要求
- Python 3.9+
- Node.js 18+
- Docker 和 Docker Compose

### 本地开发环境
1. **克隆仓库**
   ```bash
   git clone https://github.com/yourusername/fitness_llm_agent_opt.git
   cd fitness_llm_agent_opt
   ```

2. **后端设置**
   ```bash
   # 安装依赖
   pip install -r requirements.txt
   
   # 设置环境变量
   export VOLCENGINE_API_TOKEN="8bc1dd1d-6634-4527-9930-d2e28c202999"
   export VOLCENGINE_API_URL="https://ark.cn-beijing.volces.com/api/v3/chat/completions"
   
   # 启动开发服务器
   cd backend
   uvicorn app:app --reload
   ```

3. **前端设置**
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

### Docker部署
使用Docker Compose一键部署整个应用：

```bash
# 启动所有服务
docker-compose up -d

# 检查服务状态
docker-compose ps

# 查看日志
docker-compose logs -f
```

## 系统配置

### 环境变量
- `VOLCENGINE_API_URL`: 火山方舟API地址
- `VOLCENGINE_API_TOKEN`: 火山方舟API令牌
- `MILVUS_HOST`: Milvus服务器地址
- `MILVUS_PORT`: Milvus服务器端口
- `SECRET_KEY`: JWT加密密钥

### Milvus配置
系统默认创建以下数据集合：
- `fitness_knowledge`: 存储健身专业知识
- `user_training_data`: 存储用户训练数据
- `conversation_memories`: 存储对话记忆

## 使用指南

1. **用户登录**: 访问登录页面，输入用户名和密码
2. **健身咨询**: 在主页聊天界面输入健身相关问题
3. **查看训练数据**: 通过侧边栏导航到"健身数据"页面
4. **历史记录**: 查看过去的对话内容与回复

## 开发拓展

### 添加新的健身知识
1. 更新 `backend/knowledge/data/fitness_data.py` 文件
2. 重新启动后端服务以加载新数据

### 自定义UI主题
修改 `frontend/tailwind.config.js` 中的颜色配置

### 新增API端点
在 `app.py` 中添加新的FastAPI路由

## 性能优化

- Milvus向量检索使用近似最近邻算法，支持高效检索
- 多阶段Docker构建减小镜像体积
- 前端资源预加载和代码分割提升加载速度

## 贡献指南

欢迎提交Pull Request或Issue反馈问题和建议。开发前请先fork本仓库并创建新分支。 