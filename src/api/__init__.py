"""
API模块

包含：
- FastAPI应用
- 路由定义
- 中间件
- 认证授权
"""

from .main import app
from .routes import router
from .middleware import setup_middleware
from .auth import AuthManager

__all__ = [
    "app",
    "router",
    "setup_middleware",
    "AuthManager"
]