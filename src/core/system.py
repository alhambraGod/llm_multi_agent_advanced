"""
多智能体系统核心管理器

负责：
- 智能体生命周期管理
- 任务分发和协调
- 系统状态监控
- 错误处理和恢复
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

from ..agents.base import BaseAgent
from ..utils.logger import get_logger
from .config import ConfigManager
from .events import EventBus
from .state import StateManager
from .orchestrator import AgentOrchestrator

logger = get_logger(__name__)

@dataclass
class SystemMetrics:
    """系统性能指标"""
    active_agents: int
    total_requests: int
    avg_response_time: float
    error_rate: float
    memory_usage: float
    cpu_usage: float
    timestamp: datetime

class MultiAgentSystem:
    """多智能体系统主控制器"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """初始化多智能体系统
        
        Args:
            config_path: 配置文件路径
        """
        self.config = ConfigManager(config_path)
        self.event_bus = EventBus()
        self.state_manager = StateManager()
        self.orchestrator = AgentOrchestrator(self.event_bus, self.state_manager)
        
        # 智能体注册表
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_status: Dict[str, str] = {}  # idle, busy, error
        
        # 系统状态
        self.is_running = False
        self.metrics = SystemMetrics(
            active_agents=0,
            total_requests=0,
            avg_response_time=0.0,
            error_rate=0.0,
            memory_usage=0.0,
            cpu_usage=0.0,
            timestamp=datetime.now()
        )
        
        # 任务队列
        self.task_queue = asyncio.Queue()
        self.result_cache: Dict[str, Any] = {}
        
        logger.info("多智能体系统初始化完成")
    
    async def start(self) -> None:
        """启动系统"""
        try:
            logger.info("启动多智能体系统...")
            
            # 初始化各个组件
            await self._initialize_components()
            
            # 注册智能体
            await self._register_agents()
            
            # 启动任务处理循环
            self.is_running = True
            asyncio.create_task(self._task_processing_loop())
            asyncio.create_task(self._metrics_collection_loop())
            
            # 发送系统启动事件
            await self.event_bus.publish("system.started", {
                "timestamp": datetime.now(),
                "agents": list(self.agents.keys())
            })
            
            logger.info(f"系统启动成功，已注册 {len(self.agents)} 个智能体")
            
        except Exception as e:
            logger.error(f"系统启动失败: {e}")
            raise
    
    async def stop(self) -> None:
        """停止系统"""
        try:
            logger.info("停止多智能体系统...")
            
            self.is_running = False
            
            # 停止所有智能体
            for agent_id, agent in self.agents.items():
                try:
                    await agent.stop()
                    logger.info(f"智能体 {agent_id} 已停止")
                except Exception as e:
                    logger.error(f"停止智能体 {agent_id} 失败: {e}")
            
            # 发送系统停止事件
            await self.event_bus.publish("system.stopped", {
                "timestamp": datetime.now()
            })
            
            logger.info("系统已停止")
            
        except Exception as e:
            logger.error(f"系统停止失败: {e}")
            raise
    
    async def register_agent(self, agent: BaseAgent) -> None:
        """注册智能体
        
        Args:
            agent: 智能体实例
        """
        try:
            agent_id = agent.agent_id
            
            if agent_id in self.agents:
                raise ValueError(f"智能体 {agent_id} 已存在")
            
            # 注册智能体
            self.agents[agent_id] = agent
            self.agent_status[agent_id] = "idle"
            
            # 初始化智能体
            await agent.initialize()
            
            # 注册事件监听器
            await self._register_agent_events(agent)
            
            logger.info(f"智能体 {agent_id} 注册成功")
            
            # 发送智能体注册事件
            await self.event_bus.publish("agent.registered", {
                "agent_id": agent_id,
                "agent_type": agent.__class__.__name__,
                "timestamp": datetime.now()
            })
            
        except Exception as e:
            logger.error(f"注册智能体失败: {e}")
            raise
    
    async def unregister_agent(self, agent_id: str) -> None:
        """注销智能体
        
        Args:
            agent_id: 智能体ID
        """
        try:
            if agent_id not in self.agents:
                raise ValueError(f"智能体 {agent_id} 不存在")
            
            agent = self.agents[agent_id]
            
            # 停止智能体
            await agent.stop()
            
            # 移除注册
            del self.agents[agent_id]
            del self.agent_status[agent_id]
            
            logger.info(f"智能体 {agent_id} 注销成功")
            
            # 发送智能体注销事件
            await self.event_bus.publish("agent.unregistered", {
                "agent_id": agent_id,
                "timestamp": datetime.now()
            })
            
        except Exception as e:
            logger.error(f"注销智能体失败: {e}")
            raise
    
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """执行任务
        
        Args:
            task: 任务描述
            
        Returns:
            任务执行结果
        """
        try:
            task_id = task.get("task_id", f"task_{datetime.now().timestamp()}")
            task_type = task.get("type", "general")
            
            logger.info(f"开始执行任务 {task_id}, 类型: {task_type}")
            
            # 检查缓存
            cache_key = self._generate_cache_key(task)
            if cache_key in self.result_cache:
                logger.info(f"任务 {task_id} 命中缓存")
                return self.result_cache[cache_key]
            
            # 任务编排
            execution_plan = await self.orchestrator.plan_execution(task)
            
            # 执行任务
            result = await self.orchestrator.execute_plan(execution_plan)
            
            # 缓存结果
            self.result_cache[cache_key] = result
            
            # 更新指标
            self.metrics.total_requests += 1
            
            logger.info(f"任务 {task_id} 执行完成")
            
            return result
            
        except Exception as e:
            logger.error(f"任务执行失败: {e}")
            self.metrics.error_rate += 1
            raise
    
    async def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态
        
        Returns:
            系统状态信息
        """
        return {
            "is_running": self.is_running,
            "agents": {
                agent_id: {
                    "status": status,
                    "type": agent.__class__.__name__,
                    "last_activity": getattr(agent, 'last_activity', None)
                }
                for agent_id, (agent, status) in 
                zip(self.agents.keys(), 
                    zip(self.agents.values(), self.agent_status.values()))
            },
            "metrics": {
                "active_agents": len([s for s in self.agent_status.values() if s != "error"]),
                "total_requests": self.metrics.total_requests,
                "avg_response_time": self.metrics.avg_response_time,
                "error_rate": self.metrics.error_rate,
                "memory_usage": self.metrics.memory_usage,
                "cpu_usage": self.metrics.cpu_usage
            },
            "timestamp": datetime.now()
        }
    
    async def _initialize_components(self) -> None:
        """初始化系统组件"""
        # 初始化事件总线
        await self.event_bus.start()
        
        # 初始化状态管理器
        await self.state_manager.initialize()
        
        # 初始化编排器
        await self.orchestrator.initialize()
    
    async def _register_agents(self) -> None:
        """注册系统智能体"""
        from ..agents.fitness_coach import FitnessCoachAgent
        from ..agents.nutrition_expert import NutritionExpertAgent
        from ..agents.mental_health import MentalHealthAgent
        from ..agents.data_analyst import DataAnalystAgent
        
        # 创建并注册智能体
        agents_config = self.config.get("agents", {})
        
        # 健身教练智能体
        if "fitness_coach" in agents_config:
            coach_agent = FitnessCoachAgent(
                agent_id="fitness_coach",
                config=agents_config["fitness_coach"]
            )
            await self.register_agent(coach_agent)
        
        # 营养师智能体
        if "nutrition_expert" in agents_config:
            nutrition_agent = NutritionExpertAgent(
                agent_id="nutrition_expert",
                config=agents_config["nutrition_expert"]
            )
            await self.register_agent(nutrition_agent)
        
        # 心理健康智能体
        if "mental_health" in agents_config:
            mental_agent = MentalHealthAgent(
                agent_id="mental_health",
                config=agents_config["mental_health"]
            )
            await self.register_agent(mental_agent)
        
        # 数据分析智能体
        if "data_analyst" in agents_config:
            analyst_agent = DataAnalystAgent(
                agent_id="data_analyst",
                config=agents_config["data_analyst"]
            )
            await self.register_agent(analyst_agent)
    
    async def _register_agent_events(self, agent: BaseAgent) -> None:
        """注册智能体事件监听器"""
        agent_id = agent.agent_id
        
        # 监听智能体状态变化
        async def on_agent_status_change(event_data):
            if event_data.get("agent_id") == agent_id:
                new_status = event_data.get("status")
                self.agent_status[agent_id] = new_status
                logger.info(f"智能体 {agent_id} 状态变更为: {new_status}")
        
        await self.event_bus.subscribe("agent.status_changed", on_agent_status_change)
    
    async def _task_processing_loop(self) -> None:
        """任务处理循环"""
        while self.is_running:
            try:
                # 处理任务队列
                if not self.task_queue.empty():
                    task = await self.task_queue.get()
                    await self.execute_task(task)
                
                await asyncio.sleep(0.1)  # 避免CPU占用过高
                
            except Exception as e:
                logger.error(f"任务处理循环错误: {e}")
                await asyncio.sleep(1)
    
    async def _metrics_collection_loop(self) -> None:
        """指标收集循环"""
        while self.is_running:
            try:
                # 收集系统指标
                import psutil
                
                self.metrics.active_agents = len([s for s in self.agent_status.values() if s != "error"])
                self.metrics.memory_usage = psutil.virtual_memory().percent
                self.metrics.cpu_usage = psutil.cpu_percent()
                self.metrics.timestamp = datetime.now()
                
                # 发送指标事件
                await self.event_bus.publish("metrics.updated", {
                    "metrics": self.metrics,
                    "timestamp": datetime.now()
                })
                
                await asyncio.sleep(30)  # 每30秒收集一次指标
                
            except Exception as e:
                logger.error(f"指标收集错误: {e}")
                await asyncio.sleep(60)
    
    def _generate_cache_key(self, task: Dict[str, Any]) -> str:
        """生成缓存键"""
        import hashlib
        import json
        
        # 移除时间戳等动态字段
        cache_data = {k: v for k, v in task.items() 
                     if k not in ["timestamp", "task_id"]}
        
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()