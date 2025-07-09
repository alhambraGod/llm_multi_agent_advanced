"""
企业级多智能体运动健身伴侣系统

主要功能：
- 多智能体协作（健身教练、营养师、心理健康顾问、数据分析师）
- 多模态支持（文本、图像、语音、视频）
- Advanced RAG + GraphRAG 知识库
- 向量数据库集成
- 提示词优化
- 企业级部署
"""

__version__ = "1.0.0"
__author__ = "Enterprise AI Team"

from .core import *
from .agents import *
from .rag import *
from .multimodal import *
from .database import *
from .api import *
from .utils import *

__all__ = [
    "MultiAgentSystem",
    "FitnessCoachAgent",
    "NutritionExpertAgent", 
    "MentalHealthAgent",
    "DataAnalystAgent",
    "RAGEngine",
    "GraphRAGEngine",
    "MultimodalProcessor",
    "VectorDatabase",
    "PromptOptimizer"
]