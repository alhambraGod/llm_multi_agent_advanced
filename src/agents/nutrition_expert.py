"""
营养师智能体

专业功能：
- 个性化营养计划制定
- 食物营养分析
- 膳食搭配建议
- 营养目标跟踪
- 食物识别和热量计算
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

from .base import BaseAgent, AgentCapability, AgentMessage
from ..utils.logger import get_logger
from ..rag.nutrition_knowledge import NutritionKnowledgeBase
from ..multimodal.food_analyzer import FoodAnalyzer
from ..database.nutrition_db import NutritionDB

logger = get_logger(__name__)

@dataclass
class NutritionPlan:
    """营养计划"""
    plan_id: str
    user_id: str
    goal: str  # 减重、增重、维持、增肌等
    daily_calories: int
    macros: Dict[str, float]  # 蛋白质、碳水、脂肪比例
    meal_schedule: Dict[str, Dict[str, Any]]  # 餐次安排
    dietary_restrictions: List[str]
    duration_days: int
    created_at: datetime
    updated_at: datetime

@dataclass
class FoodAnalysis:
    """食物分析结果"""
    food_name: str
    portion_size: float
    calories: float
    macronutrients: Dict[str, float]
    micronutrients: Dict[str, float]
    health_score: float
    allergens: List[str]
    dietary_tags: List[str]  # 素食、无麸质等

class NutritionExpertAgent(BaseAgent):
    """营养师智能体"""
    
    def __init__(self, agent_id: str, config: Dict[str, Any] = None):
        super().__init__(
            agent_id=agent_id,
            name="AI营养师",
            description="专业的AI营养师，提供个性化营养指导和膳食建议",
            config=config
        )
        
        # 专业组件
        self.knowledge_base: Optional[NutritionKnowledgeBase] = None
        self.food_analyzer: Optional[FoodAnalyzer] = None
        self.nutrition_db: Optional[NutritionDB] = None
        
        # 营养数据
        self.nutrition_plans: Dict[str, NutritionPlan] = {}
        self.food_database: Dict[str, Dict[str, Any]] = {}
        self.user_dietary_logs: Dict[str, List[Dict[str, Any]]] = {}
        
        # 配置参数
        self.calorie_accuracy_threshold = config.get("calorie_accuracy", 0.9)
        self.macro_tolerance = config.get("macro_tolerance", 0.1)
        
    async def _agent_initialize(self) -> None:
        """初始化营养师智能体"""
        try:
            # 初始化营养知识库
            self.knowledge_base = NutritionKnowledgeBase()
            await self.knowledge_base.initialize()
            
            # 初始化食物分析器
            self.food_analyzer = FoodAnalyzer()
            await self.food_analyzer.initialize()
            
            # 初始化营养数据库
            self.nutrition_db = NutritionDB()
            await self.nutrition_db.initialize()
            
            # 加载食物数据库
            await self._load_food_database()
            
            logger.info(f"营养师智能体 {self.agent_id} 初始化完成")
            
        except Exception as e:
            logger.error(f"营养师智能体初始化失败: {e}")
            raise
    
    async def _agent_stop(self) -> None:
        """停止营养师智能体"""
        if self.nutrition_db:
            await self.nutrition_db.close()
    
    async def _register_capabilities(self) -> None:
        """注册营养师能力"""
        capabilities = [
            AgentCapability(
                name="create_nutrition_plan",
                description="创建个性化营养计划",
                input_types=["user_profile", "nutrition_goals"],
                output_types=["nutrition_plan"]
            ),
            AgentCapability(
                name="analyze_food",
                description="分析食物营养成分",
                input_types=["food_image", "food_name"],
                output_types=["nutrition_analysis"]
            ),
            AgentCapability(
                name="track_nutrition",
                description="跟踪营养摄入",
                input_types=["meal_data"],
                output_types=["nutrition_summary", "recommendations"]
            ),
            AgentCapability(
                name="suggest_meals",
                description="推荐膳食搭配",
                input_types=["dietary_preferences", "nutrition_targets"],
                output_types=["meal_suggestions"]
            ),
            AgentCapability(
                name="assess_dietary_habits",
                description="评估饮食习惯",
                input_types=["dietary_log"],
                output_types=["habit_analysis", "improvement_suggestions"]
            )
        ]
        
        for capability in capabilities:
            self.register_capability(capability)
    
    async def _process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """处理营养师请求"""
        request_type = request.get("type")
        
        try:
            if request_type == "create_nutrition_plan":
                return await self._create_nutrition_plan(request)
            
            elif request_type == "analyze_food":
                return await self._analyze_food(request)
            
            elif request_type == "track_nutrition":
                return await self._track_nutrition(request)
            
            elif request_type == "suggest_meals":
                return await self._suggest_meals(request)
            
            elif request_type == "assess_habits":
                return await self._assess_dietary_habits(request)
            
            elif request_type == "chat":
                return await self._handle_nutrition_chat(request)
            
            else:
                return {
                    "success": False,
                    "error": f"不支持的请求类型: {request_type}",
                    "supported_types": [
                        "create_nutrition_plan", "analyze_food", 
                        "track_nutrition", "suggest_meals", "assess_habits", "chat"
                    ]
                }
                
        except Exception as e:
            logger.error(f"营养师处理请求失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now()
            }
    
    async def _create_nutrition_plan(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """创建个性化营养计划"""
        user_id = request.get("user_id")
        goals = request.get("goals", [])
        dietary_restrictions = request.get("dietary_restrictions", [])
        activity_level = request.get("activity_level", "moderate")
        duration_days = request.get("duration_days", 30)
        
        try:
            # 获取用户基础信息
            user_profile = await self._get_user_nutrition_profile(user_id)
            
            # 计算营养需求
            nutrition_requirements = await self._calculate_nutrition_requirements(
                user_profile, goals, activity_level
            )
            
            # 生成营养计划
            nutrition_plan = await self._generate_nutrition_plan(
                user_id, nutrition_requirements, dietary_restrictions, duration_days
            )
            
            # 保存计划
            self.nutrition_plans[nutrition_plan.plan_id] = nutrition_plan
            
            # 生成详细说明
            plan_explanation = await self._generate_plan_explanation(nutrition_plan)
            
            return {
                "success": True,
                "plan_id": nutrition_plan.plan_id,
                "nutrition_plan": {
                    "goal": nutrition_plan.goal,
                    "daily_calories": nutrition_plan.daily_calories,
                    "macros": nutrition_plan.macros,
                    "meal_schedule": nutrition_plan.meal_schedule,
                    "dietary_restrictions": nutrition_plan.dietary_restrictions,
                    "duration_days": nutrition_plan.duration_days
                },
                "explanation": plan_explanation,
                "shopping_list": await self._generate_shopping_list(nutrition_plan),
                "meal_prep_tips": await self._get_meal_prep_tips(nutrition_plan),
                "created_at": nutrition_plan.created_at
            }
            
        except Exception as e:
            logger.error(f"创建营养计划失败: {e}")
            raise
    
    async def _analyze_food(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """分析食物营养成分"""
        food_data = request.get("food_data")  # 图像或文本描述
        portion_size = request.get("portion_size", 100)  # 克
        analysis_type = request.get("analysis_type", "comprehensive")
        
        try:
            # 识别食物
            if isinstance(food_data, str):
                # 文本描述
                food_info = await self._identify_food_by_text(food_data)
            else:
                # 图像识别
                food_info = await self.food_analyzer.analyze_food_image(food_data)
            
            if not food_info:
                return {
                    "success": False,
                    "error": "无法识别食物，请提供更清晰的描述或图像"
                }
            
            # 获取营养信息
            nutrition_data = await self._get_nutrition_data(
                food_info["name"], portion_size
            )
            
            # 生成分析结果
            food_analysis = FoodAnalysis(
                food_name=food_info["name"],
                portion_size=portion_size,
                calories=nutrition_data["calories"],
                macronutrients=nutrition_data["macronutrients"],
                micronutrients=nutrition_data["micronutrients"],
                health_score=await self._calculate_health_score(nutrition_data),
                allergens=nutrition_data.get("allergens", []),
                dietary_tags=nutrition_data.get("dietary_tags", [])
            )
            
            # 生成健康建议
            health_recommendations = await self._generate_food_recommendations(
                food_analysis
            )
            
            return {
                "success": True,
                "food_analysis": {
                    "food_name": food_analysis.food_name,
                    "portion_size": food_analysis.portion_size,
                    "calories": food_analysis.calories,
                    "macronutrients": food_analysis.macronutrients,
                    "micronutrients": food_analysis.micronutrients,
                    "health_score": food_analysis.health_score,
                    "allergens": food_analysis.allergens,
                    "dietary_tags": food_analysis.dietary_tags
                },
                "health_recommendations": health_recommendations,
                "alternative_foods": await self._suggest_healthier_alternatives(food_analysis),
                "portion_guidance": await self._get_portion_guidance(food_analysis),
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"食物分析失败: {e}")
            raise
    
    # 更多方法实现...
    async def _load_food_database(self):
        """加载食物数据库"""
        # 实现食物数据库加载逻辑
        pass
    
    async def _calculate_nutrition_requirements(self, user_profile, goals, activity_level):
        """计算营养需求"""
        # 实现营养需求计算逻辑
        pass
    
    # ... 其他方法实现