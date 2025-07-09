"""
缓存数据库模块

支持多种缓存后端：
- Redis
- Memcached
- 内存缓存
"""

import json
import pickle
import time
import hashlib
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from threading import Lock
from collections import OrderedDict

try:
    import redis
    from redis.sentinel import Sentinel
except ImportError:
    redis = None
    Sentinel = None

try:
    import pymemcache
    from pymemcache.client.base import Client as MemcacheClient
except ImportError:
    pymemcache = None
    MemcacheClient = None

from ..utils.logger import get_logger
from ..utils.config import get_config

logger = get_logger(__name__)

@dataclass
class CacheConfig:
    """缓存配置"""
    cache_type: str  # redis, memcached, memory
    host: str = "localhost"
    port: int = 6379
    password: Optional[str] = None
    db: int = 0
    max_connections: int = 50
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    retry_on_timeout: bool = True
    health_check_interval: int = 30
    # 内存缓存配置
    max_size: int = 1000
    ttl: int = 3600  # 默认TTL（秒）
    # Redis集群配置
    cluster_nodes: Optional[List[Dict[str, Union[str, int]]]] = None
    # Sentinel配置
    sentinel_hosts: Optional[List[tuple]] = None
    sentinel_service: Optional[str] = None
    
class MemoryCache:
    """
    内存缓存实现（LRU）
    """
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache = OrderedDict()
        self.expiry = {}
        self.lock = Lock()
        
    def _is_expired(self, key: str) -> bool:
        """检查键是否过期"""
        if key not in self.expiry:
            return False
        return time.time() > self.expiry[key]
        
    def _cleanup_expired(self):
        """清理过期键"""
        current_time = time.time()
        expired_keys = [
            key for key, expiry_time in self.expiry.items()
            if current_time > expiry_time
        ]
        for key in expired_keys:
            self.cache.pop(key, None)
            self.expiry.pop(key, None)
            
    def get(self, key: str) -> Any:
        """获取缓存值"""
        with self.lock:
            if self._is_expired(key):
                self.cache.pop(key, None)
                self.expiry.pop(key, None)
                return None
                
            if key in self.cache:
                # 移到末尾（最近使用）
                value = self.cache.pop(key)
                self.cache[key] = value
                return value
            return None
            
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存值"""
        with self.lock:
            # 清理过期键
            self._cleanup_expired()
            
            # 如果缓存已满，删除最旧的项
            if len(self.cache) >= self.max_size and key not in self.cache:
                oldest_key = next(iter(self.cache))
                self.cache.pop(oldest_key)
                self.expiry.pop(oldest_key, None)
                
            self.cache[key] = value
            
            # 设置过期时间
            if ttl is None:
                ttl = self.default_ttl
            if ttl > 0:
                self.expiry[key] = time.time() + ttl
            else:
                self.expiry.pop(key, None)
                
            return True
            
    def delete(self, key: str) -> bool:
        """删除缓存值"""
        with self.lock:
            if key in self.cache:
                self.cache.pop(key)
                self.expiry.pop(key, None)
                return True
            return False
            
    def clear(self) -> bool:
        """清空缓存"""
        with self.lock:
            self.cache.clear()
            self.expiry.clear()
            return True
            
    def keys(self) -> List[str]:
        """获取所有键"""
        with self.lock:
            self._cleanup_expired()
            return list(self.cache.keys())
            
    def size(self) -> int:
        """获取缓存大小"""
        with self.lock:
            self._cleanup_expired()
            return len(self.cache)
            
class CacheDatabase:
    """
    缓存数据库管理器
    
    支持多种缓存后端和高级功能
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        """
        初始化缓存数据库
        
        Args:
            config: 缓存配置
        """
        self.config = config or self._load_config()
        self.client = None
        self.metrics = {
            "total_operations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "cache_sets": 0,
            "cache_deletes": 0,
            "total_time": 0.0,
            "avg_operation_time": 0.0
        }
        
        # 初始化缓存客户端
        self._initialize_client()
        
    def _load_config(self) -> CacheConfig:
        """加载缓存配置"""
        config = get_config()
        cache_config = config.get("database", {}).get("cache", {})
        
        return CacheConfig(
            cache_type=cache_config.get("type", "memory"),
            host=cache_config.get("host", "localhost"),
            port=cache_config.get("port", 6379),
            password=cache_config.get("password"),
            db=cache_config.get("db", 0),
            max_connections=cache_config.get("max_connections", 50),
            socket_timeout=cache_config.get("socket_timeout", 5.0),
            socket_connect_timeout=cache_config.get("socket_connect_timeout", 5.0),
            retry_on_timeout=cache_config.get("retry_on_timeout", True),
            health_check_interval=cache_config.get("health_check_interval", 30),
            max_size=cache_config.get("max_size", 1000),
            ttl=cache_config.get("ttl", 3600),
            cluster_nodes=cache_config.get("cluster_nodes"),
            sentinel_hosts=cache_config.get("sentinel_hosts"),
            sentinel_service=cache_config.get("sentinel_service")
        )
        
    def _initialize_client(self):
        """初始化缓存客户端"""
        try:
            if self.config.cache_type == "redis":
                self._init_redis()
            elif self.config.cache_type == "memcached":
                self._init_memcached()
            elif self.config.cache_type == "memory":
                self._init_memory()
            else:
                raise ValueError(f"不支持的缓存类型: {self.config.cache_type}")
                
            logger.info(f"缓存客户端初始化成功: {self.config.cache_type}")
            
        except Exception as e:
            logger.error(f"缓存客户端初始化失败: {e}")
            raise
            
    def _init_redis(self):
        """初始化Redis客户端"""
        if redis is None:
            raise ImportError("请安装redis: pip install redis")
            
        if self.config.cluster_nodes:
            # Redis集群模式
            from redis.cluster import RedisCluster
            self.client = RedisCluster(
                startup_nodes=self.config.cluster_nodes,
                password=self.config.password,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                retry_on_timeout=self.config.retry_on_timeout,
                health_check_interval=self.config.health_check_interval
            )
        elif self.config.sentinel_hosts:
            # Redis Sentinel模式
            sentinel = Sentinel(self.config.sentinel_hosts)
            self.client = sentinel.master_for(
                self.config.sentinel_service,
                password=self.config.password,
                db=self.config.db,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                retry_on_timeout=self.config.retry_on_timeout
            )
        else:
            # 单机模式
            connection_pool = redis.ConnectionPool(
                host=self.config.host,
                port=self.config.port,
                password=self.config.password,
                db=self.config.db,
                max_connections=self.config.max_connections,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                retry_on_timeout=self.config.retry_on_timeout,
                health_check_interval=self.config.health_check_interval
            )
            self.client = redis.Redis(connection_pool=connection_pool)
            
    def _init_memcached(self):
        """初始化Memcached客户端"""
        if pymemcache is None:
            raise ImportError("请安装pymemcache: pip install pymemcache")
            
        self.client = MemcacheClient(
            (self.config.host, self.config.port),
            timeout=self.config.socket_timeout,
            connect_timeout=self.config.socket_connect_timeout
        )
        
    def _init_memory(self):
        """初始化内存缓存"""
        self.client = MemoryCache(
            max_size=self.config.max_size,
            default_ttl=self.config.ttl
        )
        
    def _serialize_value(self, value: Any) -> Union[str, bytes]:
        """序列化值"""
        if isinstance(value, (str, int, float, bool)):
            return json.dumps(value)
        else:
            return pickle.dumps(value)
            
    def _deserialize_value(self, value: Union[str, bytes]) -> Any:
        """反序列化值"""
        if value is None:
            return None
            
        try:
            # 尝试JSON反序列化
            if isinstance(value, bytes):
                value = value.decode('utf-8')
            return json.loads(value)
        except (json.JSONDecodeError, UnicodeDecodeError):
            # 使用pickle反序列化
            if isinstance(value, str):
                value = value.encode('utf-8')
            return pickle.loads(value)
            
    def _generate_key(self, key: str, prefix: str = "") -> str:
        """生成缓存键"""
        if prefix:
            return f"{prefix}:{key}"
        return key
        
    def get(self, key: str, prefix: str = "") -> Any:
        """
        获取缓存值
        
        Args:
            key: 缓存键
            prefix: 键前缀
            
        Returns:
            缓存值，不存在返回None
        """
        start_time = time.time()
        cache_key = self._generate_key(key, prefix)
        
        try:
            if self.config.cache_type == "redis":
                value = self.client.get(cache_key)
            elif self.config.cache_type == "memcached":
                value = self.client.get(cache_key)
            else:  # memory
                value = self.client.get(cache_key)
                
            # 更新指标
            self.metrics["total_operations"] += 1
            if value is not None:
                self.metrics["cache_hits"] += 1
                result = self._deserialize_value(value)
            else:
                self.metrics["cache_misses"] += 1
                result = None
                
            operation_time = time.time() - start_time
            self.metrics["total_time"] += operation_time
            self.metrics["avg_operation_time"] = self.metrics["total_time"] / self.metrics["total_operations"]
            
            return result
            
        except Exception as e:
            self.metrics["total_operations"] += 1
            self.metrics["cache_misses"] += 1
            logger.error(f"获取缓存失败 {cache_key}: {e}")
            return None
            
    def set(self, key: str, value: Any, ttl: Optional[int] = None, prefix: str = "") -> bool:
        """
        设置缓存值
        
        Args:
            key: 缓存键
            value: 缓存值
            ttl: 过期时间（秒）
            prefix: 键前缀
            
        Returns:
            是否设置成功
        """
        start_time = time.time()
        cache_key = self._generate_key(key, prefix)
        
        try:
            serialized_value = self._serialize_value(value)
            
            if self.config.cache_type == "redis":
                if ttl is not None:
                    result = self.client.setex(cache_key, ttl, serialized_value)
                else:
                    result = self.client.set(cache_key, serialized_value)
            elif self.config.cache_type == "memcached":
                expire = ttl or self.config.ttl
                result = self.client.set(cache_key, serialized_value, expire=expire)
            else:  # memory
                result = self.client.set(cache_key, serialized_value, ttl)
                
            # 更新指标
            self.metrics["total_operations"] += 1
            self.metrics["cache_sets"] += 1
            operation_time = time.time() - start_time
            self.metrics["total_time"] += operation_time
            self.metrics["avg_operation_time"] = self.metrics["total_time"] / self.metrics["total_operations"]
            
            return bool(result)
            
        except Exception as e:
            self.metrics["total_operations"] += 1
            logger.error(f"设置缓存失败 {cache_key}: {e}")
            return False
            
    def delete(self, key: str, prefix: str = "") -> bool:
        """
        删除缓存值
        
        Args:
            key: 缓存键
            prefix: 键前缀
            
        Returns:
            是否删除成功
        """
        start_time = time.time()
        cache_key = self._generate_key(key, prefix)
        
        try:
            if self.config.cache_type == "redis":
                result = self.client.delete(cache_key)
            elif self.config.cache_type == "memcached":
                result = self.client.delete(cache_key)
            else:  # memory
                result = self.client.delete(cache_key)
                
            # 更新指标
            self.metrics["total_operations"] += 1
            self.metrics["cache_deletes"] += 1
            operation_time = time.time() - start_time
            self.metrics["total_time"] += operation_time
            self.metrics["avg_operation_time"] = self.metrics["total_time"] / self.metrics["total_operations"]
            
            return bool(result)
            
        except Exception as e:
            self.metrics["total_operations"] += 1
            logger.error(f"删除缓存失败 {cache_key}: {e}")
            return False
            
    def exists(self, key: str, prefix: str = "") -> bool:
        """
        检查缓存键是否存在
        
        Args:
            key: 缓存键
            prefix: 键前缀
            
        Returns:
            是否存在
        """
        cache_key = self._generate_key(key, prefix)
        
        try:
            if self.config.cache_type == "redis":
                return bool(self.client.exists(cache_key))
            elif self.config.cache_type == "memcached":
                return self.client.get(cache_key) is not None
            else:  # memory
                return self.client.get(cache_key) is not None
                
        except Exception as e:
            logger.error(f"检查缓存存在性失败 {cache_key}: {e}")
            return False
            
    def expire(self, key: str, ttl: int, prefix: str = "") -> bool:
        """
        设置缓存过期时间
        
        Args:
            key: 缓存键
            ttl: 过期时间（秒）
            prefix: 键前缀
            
        Returns:
            是否设置成功
        """
        cache_key = self._generate_key(key, prefix)
        
        try:
            if self.config.cache_type == "redis":
                return bool(self.client.expire(cache_key, ttl))
            else:
                # Memcached和内存缓存不支持单独设置过期时间
                logger.warning(f"缓存类型 {self.config.cache_type} 不支持设置过期时间")
                return False
                
        except Exception as e:
            logger.error(f"设置缓存过期时间失败 {cache_key}: {e}")
            return False
            
    def clear(self, prefix: str = "") -> bool:
        """
        清空缓存
        
        Args:
            prefix: 键前缀，如果指定则只清空该前缀的键
            
        Returns:
            是否清空成功
        """
        try:
            if self.config.cache_type == "redis":
                if prefix:
                    # 删除指定前缀的键
                    pattern = f"{prefix}:*"
                    keys = self.client.keys(pattern)
                    if keys:
                        return bool(self.client.delete(*keys))
                    return True
                else:
                    return bool(self.client.flushdb())
            elif self.config.cache_type == "memcached":
                return bool(self.client.flush_all())
            else:  # memory
                return self.client.clear()
                
        except Exception as e:
            logger.error(f"清空缓存失败: {e}")
            return False
            
    def get_keys(self, pattern: str = "*", prefix: str = "") -> List[str]:
        """
        获取匹配的缓存键
        
        Args:
            pattern: 匹配模式
            prefix: 键前缀
            
        Returns:
            匹配的键列表
        """
        try:
            if prefix:
                pattern = f"{prefix}:{pattern}"
                
            if self.config.cache_type == "redis":
                keys = self.client.keys(pattern)
                return [key.decode('utf-8') if isinstance(key, bytes) else key for key in keys]
            elif self.config.cache_type == "memory":
                import fnmatch
                all_keys = self.client.keys()
                return [key for key in all_keys if fnmatch.fnmatch(key, pattern)]
            else:
                # Memcached不支持键列表
                logger.warning("Memcached不支持获取键列表")
                return []
                
        except Exception as e:
            logger.error(f"获取缓存键失败: {e}")
            return []
            
    def mget(self, keys: List[str], prefix: str = "") -> Dict[str, Any]:
        """
        批量获取缓存值
        
        Args:
            keys: 缓存键列表
            prefix: 键前缀
            
        Returns:
            键值对字典
        """
        cache_keys = [self._generate_key(key, prefix) for key in keys]
        result = {}
        
        try:
            if self.config.cache_type == "redis":
                values = self.client.mget(cache_keys)
                for i, value in enumerate(values):
                    if value is not None:
                        result[keys[i]] = self._deserialize_value(value)
            elif self.config.cache_type == "memcached":
                cache_dict = self.client.get_many(cache_keys)
                for cache_key, value in cache_dict.items():
                    original_key = cache_key.replace(f"{prefix}:", "") if prefix else cache_key
                    result[original_key] = self._deserialize_value(value)
            else:  # memory
                for i, cache_key in enumerate(cache_keys):
                    value = self.client.get(cache_key)
                    if value is not None:
                        result[keys[i]] = value
                        
            # 更新指标
            self.metrics["total_operations"] += len(keys)
            self.metrics["cache_hits"] += len(result)
            self.metrics["cache_misses"] += len(keys) - len(result)
            
            return result
            
        except Exception as e:
            logger.error(f"批量获取缓存失败: {e}")
            return {}
            
    def mset(self, data: Dict[str, Any], ttl: Optional[int] = None, prefix: str = "") -> bool:
        """
        批量设置缓存值
        
        Args:
            data: 键值对字典
            ttl: 过期时间（秒）
            prefix: 键前缀
            
        Returns:
            是否设置成功
        """
        try:
            if self.config.cache_type == "redis":
                # 序列化所有值
                serialized_data = {}
                for key, value in data.items():
                    cache_key = self._generate_key(key, prefix)
                    serialized_data[cache_key] = self._serialize_value(value)
                    
                result = self.client.mset(serialized_data)
                
                # 如果指定了TTL，需要单独设置过期时间
                if ttl is not None and result:
                    for cache_key in serialized_data.keys():
                        self.client.expire(cache_key, ttl)
                        
            elif self.config.cache_type == "memcached":
                serialized_data = {}
                for key, value in data.items():
                    cache_key = self._generate_key(key, prefix)
                    serialized_data[cache_key] = self._serialize_value(value)
                    
                expire = ttl or self.config.ttl
                result = self.client.set_many(serialized_data, expire=expire)
                
            else:  # memory
                result = True
                for key, value in data.items():
                    cache_key = self._generate_key(key, prefix)
                    if not self.client.set(cache_key, value, ttl):
                        result = False
                        
            # 更新指标
            self.metrics["total_operations"] += len(data)
            self.metrics["cache_sets"] += len(data)
            
            return bool(result)
            
        except Exception as e:
            logger.error(f"批量设置缓存失败: {e}")
            return False
            
    def cache_decorator(self, ttl: Optional[int] = None, prefix: str = "", 
                       key_func: Optional[Callable] = None):
        """
        缓存装饰器
        
        Args:
            ttl: 过期时间（秒）
            prefix: 键前缀
            key_func: 自定义键生成函数
            
        Returns:
            装饰器函数
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                # 生成缓存键
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    # 使用函数名和参数生成键
                    args_str = str(args) + str(sorted(kwargs.items()))
                    cache_key = f"{func.__name__}:{hashlib.md5(args_str.encode()).hexdigest()}"
                    
                # 尝试从缓存获取
                result = self.get(cache_key, prefix)
                if result is not None:
                    return result
                    
                # 执行函数并缓存结果
                result = func(*args, **kwargs)
                self.set(cache_key, result, ttl, prefix)
                return result
                
            return wrapper
        return decorator
        
    def get_metrics(self) -> Dict[str, Any]:
        """
        获取缓存指标
        
        Returns:
            指标字典
        """
        hit_rate = 0.0
        if self.metrics["cache_hits"] + self.metrics["cache_misses"] > 0:
            hit_rate = self.metrics["cache_hits"] / (self.metrics["cache_hits"] + self.metrics["cache_misses"]) * 100
            
        return {
            **self.metrics,
            "cache_hit_rate": hit_rate,
            "cache_type": self.config.cache_type,
            "cache_size": self._get_cache_size()
        }
        
    def _get_cache_size(self) -> int:
        """获取缓存大小"""
        try:
            if self.config.cache_type == "redis":
                return self.client.dbsize()
            elif self.config.cache_type == "memory":
                return self.client.size()
            else:
                return -1  # Memcached不支持获取大小
        except Exception:
            return -1
            
    def health_check(self) -> Dict[str, Any]:
        """
        健康检查
        
        Returns:
            健康状态字典
        """
        try:
            start_time = time.time()
            
            # 执行简单的设置和获取操作
            test_key = "health_check"
            test_value = "ok"
            
            self.set(test_key, test_value, ttl=60)
            result = self.get(test_key)
            self.delete(test_key)
            
            response_time = time.time() - start_time
            
            if result == test_value:
                return {
                    "status": "healthy",
                    "cache_type": self.config.cache_type,
                    "response_time": response_time,
                    "cache_size": self._get_cache_size(),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "status": "unhealthy",
                    "cache_type": self.config.cache_type,
                    "error": "缓存读写测试失败",
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            return {
                "status": "unhealthy",
                "cache_type": self.config.cache_type,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            
    def close(self):
        """关闭缓存连接"""
        try:
            if self.config.cache_type == "redis" and hasattr(self.client, 'close'):
                self.client.close()
            elif self.config.cache_type == "memcached" and hasattr(self.client, 'close'):
                self.client.close()
            logger.info("缓存连接已关闭")
        except Exception as e:
            logger.error(f"关闭缓存连接失败: {e}")