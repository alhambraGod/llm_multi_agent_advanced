"""
关系数据库模块

支持多种关系数据库：
- PostgreSQL
- MySQL
- SQLite
- SQL Server
"""

import logging
import time
from typing import Dict, List, Any, Optional, Union
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    from psycopg2.pool import ThreadedConnectionPool
except ImportError:
    psycopg2 = None

try:
    import pymysql
    from pymysql.cursors import DictCursor
except ImportError:
    pymysql = None

try:
    import sqlite3
except ImportError:
    sqlite3 = None

try:
    import pyodbc
except ImportError:
    pyodbc = None

from ..utils.logger import get_logger
from ..utils.config import get_config

logger = get_logger(__name__)

@dataclass
class DatabaseConfig:
    """数据库配置"""
    db_type: str  # postgresql, mysql, sqlite, sqlserver
    host: Optional[str] = None
    port: Optional[int] = None
    database: str = ""
    username: Optional[str] = None
    password: Optional[str] = None
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600
    ssl_mode: str = "prefer"
    charset: str = "utf8mb4"
    
class RelationalDatabase:
    """
    关系数据库管理器
    
    支持多种数据库类型和连接池管理
    """
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        """
        初始化关系数据库
        
        Args:
            config: 数据库配置
        """
        self.config = config or self._load_config()
        self.pool = None
        self.connection = None
        self.metrics = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "total_time": 0.0,
            "avg_query_time": 0.0
        }
        
        # 初始化数据库连接
        self._initialize_connection()
        
    def _load_config(self) -> DatabaseConfig:
        """加载数据库配置"""
        config = get_config()
        db_config = config.get("database", {}).get("relational", {})
        
        return DatabaseConfig(
            db_type=db_config.get("type", "sqlite"),
            host=db_config.get("host", "localhost"),
            port=db_config.get("port"),
            database=db_config.get("database", "fitness_app.db"),
            username=db_config.get("username"),
            password=db_config.get("password"),
            pool_size=db_config.get("pool_size", 10),
            max_overflow=db_config.get("max_overflow", 20),
            pool_timeout=db_config.get("pool_timeout", 30),
            pool_recycle=db_config.get("pool_recycle", 3600),
            ssl_mode=db_config.get("ssl_mode", "prefer"),
            charset=db_config.get("charset", "utf8mb4")
        )
        
    def _initialize_connection(self):
        """初始化数据库连接"""
        try:
            if self.config.db_type == "postgresql":
                self._init_postgresql()
            elif self.config.db_type == "mysql":
                self._init_mysql()
            elif self.config.db_type == "sqlite":
                self._init_sqlite()
            elif self.config.db_type == "sqlserver":
                self._init_sqlserver()
            else:
                raise ValueError(f"不支持的数据库类型: {self.config.db_type}")
                
            logger.info(f"数据库连接初始化成功: {self.config.db_type}")
            
        except Exception as e:
            logger.error(f"数据库连接初始化失败: {e}")
            raise
            
    def _init_postgresql(self):
        """初始化PostgreSQL连接"""
        if psycopg2 is None:
            raise ImportError("请安装psycopg2: pip install psycopg2-binary")
            
        connection_params = {
            "host": self.config.host,
            "port": self.config.port or 5432,
            "database": self.config.database,
            "user": self.config.username,
            "password": self.config.password,
            "sslmode": self.config.ssl_mode
        }
        
        # 创建连接池
        self.pool = ThreadedConnectionPool(
            minconn=1,
            maxconn=self.config.pool_size,
            **connection_params
        )
        
    def _init_mysql(self):
        """初始化MySQL连接"""
        if pymysql is None:
            raise ImportError("请安装pymysql: pip install pymysql")
            
        self.connection_params = {
            "host": self.config.host,
            "port": self.config.port or 3306,
            "database": self.config.database,
            "user": self.config.username,
            "password": self.config.password,
            "charset": self.config.charset,
            "cursorclass": DictCursor,
            "autocommit": True
        }
        
    def _init_sqlite(self):
        """初始化SQLite连接"""
        if sqlite3 is None:
            raise ImportError("SQLite3 不可用")
            
        self.db_path = self.config.database
        
    def _init_sqlserver(self):
        """初始化SQL Server连接"""
        if pyodbc is None:
            raise ImportError("请安装pyodbc: pip install pyodbc")
            
        self.connection_string = (
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={self.config.host},{self.config.port or 1433};"
            f"DATABASE={self.config.database};"
            f"UID={self.config.username};"
            f"PWD={self.config.password}"
        )
        
    @contextmanager
    def get_connection(self):
        """获取数据库连接"""
        connection = None
        try:
            if self.config.db_type == "postgresql":
                connection = self.pool.getconn()
            elif self.config.db_type == "mysql":
                connection = pymysql.connect(**self.connection_params)
            elif self.config.db_type == "sqlite":
                connection = sqlite3.connect(self.db_path)
                connection.row_factory = sqlite3.Row
            elif self.config.db_type == "sqlserver":
                connection = pyodbc.connect(self.connection_string)
                
            yield connection
            
        except Exception as e:
            if connection:
                connection.rollback()
            raise e
        finally:
            if connection:
                if self.config.db_type == "postgresql":
                    self.pool.putconn(connection)
                else:
                    connection.close()
                    
    def execute_query(self, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """
        执行查询语句
        
        Args:
            query: SQL查询语句
            params: 查询参数
            
        Returns:
            查询结果列表
        """
        start_time = time.time()
        
        try:
            with self.get_connection() as conn:
                if self.config.db_type == "postgresql":
                    cursor = conn.cursor(cursor_factory=RealDictCursor)
                elif self.config.db_type in ["mysql", "sqlserver"]:
                    cursor = conn.cursor()
                else:  # sqlite
                    cursor = conn.cursor()
                    
                cursor.execute(query, params or ())
                
                if query.strip().upper().startswith("SELECT"):
                    results = cursor.fetchall()
                    if self.config.db_type == "sqlite":
                        results = [dict(row) for row in results]
                    elif self.config.db_type == "postgresql":
                        results = [dict(row) for row in results]
                    else:
                        columns = [desc[0] for desc in cursor.description]
                        results = [dict(zip(columns, row)) for row in results]
                else:
                    conn.commit()
                    results = [{"affected_rows": cursor.rowcount}]
                    
                cursor.close()
                
                # 更新指标
                self.metrics["total_queries"] += 1
                self.metrics["successful_queries"] += 1
                query_time = time.time() - start_time
                self.metrics["total_time"] += query_time
                self.metrics["avg_query_time"] = self.metrics["total_time"] / self.metrics["total_queries"]
                
                logger.debug(f"查询执行成功，耗时: {query_time:.3f}s")
                return results
                
        except Exception as e:
            self.metrics["total_queries"] += 1
            self.metrics["failed_queries"] += 1
            logger.error(f"查询执行失败: {e}")
            raise
            
    def execute_many(self, query: str, params_list: List[tuple]) -> int:
        """
        批量执行语句
        
        Args:
            query: SQL语句
            params_list: 参数列表
            
        Returns:
            影响的行数
        """
        start_time = time.time()
        
        try:
            with self.get_connection() as conn:
                if self.config.db_type == "postgresql":
                    cursor = conn.cursor()
                elif self.config.db_type in ["mysql", "sqlserver"]:
                    cursor = conn.cursor()
                else:  # sqlite
                    cursor = conn.cursor()
                    
                cursor.executemany(query, params_list)
                affected_rows = cursor.rowcount
                conn.commit()
                cursor.close()
                
                # 更新指标
                self.metrics["total_queries"] += 1
                self.metrics["successful_queries"] += 1
                query_time = time.time() - start_time
                self.metrics["total_time"] += query_time
                self.metrics["avg_query_time"] = self.metrics["total_time"] / self.metrics["total_queries"]
                
                logger.debug(f"批量执行成功，影响行数: {affected_rows}，耗时: {query_time:.3f}s")
                return affected_rows
                
        except Exception as e:
            self.metrics["total_queries"] += 1
            self.metrics["failed_queries"] += 1
            logger.error(f"批量执行失败: {e}")
            raise
            
    def create_table(self, table_name: str, columns: Dict[str, str], 
                    primary_key: Optional[str] = None, 
                    indexes: Optional[List[str]] = None) -> bool:
        """
        创建表
        
        Args:
            table_name: 表名
            columns: 列定义字典 {列名: 类型}
            primary_key: 主键列名
            indexes: 索引列名列表
            
        Returns:
            是否创建成功
        """
        try:
            # 构建列定义
            column_defs = []
            for col_name, col_type in columns.items():
                column_defs.append(f"{col_name} {col_type}")
                
            if primary_key:
                column_defs.append(f"PRIMARY KEY ({primary_key})")
                
            columns_sql = ", ".join(column_defs)
            
            # 创建表
            create_sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns_sql})"
            self.execute_query(create_sql)
            
            # 创建索引
            if indexes:
                for index_col in indexes:
                    index_name = f"idx_{table_name}_{index_col}"
                    index_sql = f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name} ({index_col})"
                    self.execute_query(index_sql)
                    
            logger.info(f"表 {table_name} 创建成功")
            return True
            
        except Exception as e:
            logger.error(f"创建表 {table_name} 失败: {e}")
            return False
            
    def insert_data(self, table_name: str, data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> int:
        """
        插入数据
        
        Args:
            table_name: 表名
            data: 数据字典或数据字典列表
            
        Returns:
            插入的行数
        """
        try:
            if isinstance(data, dict):
                data = [data]
                
            if not data:
                return 0
                
            # 构建插入语句
            columns = list(data[0].keys())
            placeholders = ", ".join(["%s"] * len(columns))
            
            if self.config.db_type == "sqlite":
                placeholders = ", ".join(["?"] * len(columns))
            elif self.config.db_type == "sqlserver":
                placeholders = ", ".join(["?"] * len(columns))
                
            insert_sql = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
            
            # 准备参数
            params_list = []
            for row in data:
                params_list.append(tuple(row[col] for col in columns))
                
            if len(params_list) == 1:
                self.execute_query(insert_sql, params_list[0])
                return 1
            else:
                return self.execute_many(insert_sql, params_list)
                
        except Exception as e:
            logger.error(f"插入数据到 {table_name} 失败: {e}")
            raise
            
    def update_data(self, table_name: str, data: Dict[str, Any], 
                   where_clause: str, where_params: Optional[tuple] = None) -> int:
        """
        更新数据
        
        Args:
            table_name: 表名
            data: 更新数据字典
            where_clause: WHERE条件
            where_params: WHERE参数
            
        Returns:
            更新的行数
        """
        try:
            # 构建更新语句
            set_clauses = []
            set_params = []
            
            for col, value in data.items():
                if self.config.db_type in ["sqlite", "sqlserver"]:
                    set_clauses.append(f"{col} = ?")
                else:
                    set_clauses.append(f"{col} = %s")
                set_params.append(value)
                
            set_sql = ", ".join(set_clauses)
            update_sql = f"UPDATE {table_name} SET {set_sql} WHERE {where_clause}"
            
            # 合并参数
            all_params = set_params + list(where_params or [])
            
            result = self.execute_query(update_sql, tuple(all_params))
            return result[0].get("affected_rows", 0)
            
        except Exception as e:
            logger.error(f"更新 {table_name} 数据失败: {e}")
            raise
            
    def delete_data(self, table_name: str, where_clause: str, 
                   where_params: Optional[tuple] = None) -> int:
        """
        删除数据
        
        Args:
            table_name: 表名
            where_clause: WHERE条件
            where_params: WHERE参数
            
        Returns:
            删除的行数
        """
        try:
            delete_sql = f"DELETE FROM {table_name} WHERE {where_clause}"
            result = self.execute_query(delete_sql, where_params)
            return result[0].get("affected_rows", 0)
            
        except Exception as e:
            logger.error(f"删除 {table_name} 数据失败: {e}")
            raise
            
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """
        获取表信息
        
        Args:
            table_name: 表名
            
        Returns:
            表信息字典
        """
        try:
            if self.config.db_type == "postgresql":
                query = """
                    SELECT column_name, data_type, is_nullable, column_default
                    FROM information_schema.columns
                    WHERE table_name = %s
                    ORDER BY ordinal_position
                """
            elif self.config.db_type == "mysql":
                query = """
                    SELECT COLUMN_NAME as column_name, DATA_TYPE as data_type, 
                           IS_NULLABLE as is_nullable, COLUMN_DEFAULT as column_default
                    FROM information_schema.COLUMNS
                    WHERE TABLE_NAME = %s
                    ORDER BY ORDINAL_POSITION
                """
            elif self.config.db_type == "sqlite":
                query = f"PRAGMA table_info({table_name})"
            else:  # sqlserver
                query = """
                    SELECT COLUMN_NAME as column_name, DATA_TYPE as data_type,
                           IS_NULLABLE as is_nullable, COLUMN_DEFAULT as column_default
                    FROM INFORMATION_SCHEMA.COLUMNS
                    WHERE TABLE_NAME = ?
                    ORDER BY ORDINAL_POSITION
                """
                
            if self.config.db_type == "sqlite":
                columns = self.execute_query(query)
                # SQLite返回格式不同，需要转换
                formatted_columns = []
                for col in columns:
                    formatted_columns.append({
                        "column_name": col["name"],
                        "data_type": col["type"],
                        "is_nullable": "YES" if col["notnull"] == 0 else "NO",
                        "column_default": col["dflt_value"]
                    })
                columns = formatted_columns
            else:
                columns = self.execute_query(query, (table_name,))
                
            return {
                "table_name": table_name,
                "columns": columns,
                "column_count": len(columns)
            }
            
        except Exception as e:
            logger.error(f"获取表 {table_name} 信息失败: {e}")
            return {}
            
    def get_metrics(self) -> Dict[str, Any]:
        """
        获取性能指标
        
        Returns:
            性能指标字典
        """
        return {
            **self.metrics,
            "success_rate": (
                self.metrics["successful_queries"] / max(self.metrics["total_queries"], 1)
            ) * 100,
            "database_type": self.config.db_type,
            "pool_size": self.config.pool_size if self.pool else 1
        }
        
    def health_check(self) -> Dict[str, Any]:
        """
        健康检查
        
        Returns:
            健康状态字典
        """
        try:
            # 执行简单查询测试连接
            if self.config.db_type == "postgresql":
                test_query = "SELECT 1 as test"
            elif self.config.db_type == "mysql":
                test_query = "SELECT 1 as test"
            elif self.config.db_type == "sqlite":
                test_query = "SELECT 1 as test"
            else:  # sqlserver
                test_query = "SELECT 1 as test"
                
            start_time = time.time()
            result = self.execute_query(test_query)
            response_time = time.time() - start_time
            
            return {
                "status": "healthy",
                "database_type": self.config.db_type,
                "response_time": response_time,
                "test_result": result[0] if result else None,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "database_type": self.config.db_type,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            
    def close(self):
        """关闭数据库连接"""
        try:
            if self.pool and self.config.db_type == "postgresql":
                self.pool.closeall()
            logger.info("数据库连接已关闭")
        except Exception as e:
            logger.error(f"关闭数据库连接失败: {e}")