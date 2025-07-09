"""
健身教练智能体

专业功能：
- 个性化训练计划制定
- 运动动作指导和纠正
- 训练进度跟踪和调整
- 运动安全监控
- 多模态运动分析（视频、图像、数据）
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import cv2
import numpy as np

from .base import BaseAgent, AgentCapability, AgentMessage
from ..utils.logger import get_logger
from ..rag.fitness_knowledge import FitnessKnowledgeBase
from ..multimodal.pose_analyzer import PoseAnalyzer
from ..database.user_profile import UserProfileDB

logger = get_logger(__name__)

@dataclass
class WorkoutPlan:
    """训练计划"""
    plan_id: str
    user_id: str
    goal: str  # 减脂、增肌、塑形、康复等
    duration_weeks: int
    exercises: List[Dict[str, Any]]
    schedule: Dict[str, List[str]]  # 周计划
    intensity_level: str  # 初级、中级、高级
    equipment_needed: List[str]
    created_at: datetime
    updated_at: datetime

@dataclass
class ExerciseAnalysis:
    """运动分析结果"""
    exercise_name: str
    form_score: float  # 动作标准度评分 0-100
    rep_count: int
    duration: float
    calories_burned: float
    feedback: List[str]  # 改进建议
    pose_keypoints: List[Tuple[float, float]]  # 关键点坐标
    risk_assessment: str  # 风险评估

class FitnessCoachAgent(BaseAgent):
    """健身教练智能体"""
    
    def __init__(self, agent_id: str, config: Dict[str, Any] = None):
        super().__init__(
            agent_id=agent_id,
            name="AI健身教练",
            description="专业的AI健身教练，提供个性化训练指导和运动分析",
            config=config
        )
        
        # 专业组件
        self.knowledge_base: Optional[FitnessKnowledgeBase] = None
        self.pose_analyzer: Optional[PoseAnalyzer] = None
        self.user_profile_db: Optional[UserProfileDB] = None
        
        # 训练数据
        self.workout_plans: Dict[str, WorkoutPlan] = {}
        self.exercise_library: Dict[str, Dict[str, Any]] = {}
        self.user_progress: Dict[str, List[Dict[str, Any]]] = {}
        
        # 配置参数
        self.max_plan_duration = config.get("max_plan_duration_weeks", 12)
        self.safety_threshold = config.get("safety_threshold", 0.8)
        self.form_threshold = config.get("form_threshold", 0.7)
        
    async def _agent_initialize(self) -> None:
        """初始化健身教练智能体"""
        try:
            # 初始化知识库
            self.knowledge_base = FitnessKnowledgeBase()
            await self.knowledge_base.initialize()
            
            # 初始化姿态分析器
            self.pose_analyzer = PoseAnalyzer()
            await self.pose_analyzer.initialize()
            
            # 初始化用户档案数据库
            self.user_profile_db = UserProfileDB()
            await self.user_profile_db.initialize()
            
            # 加载运动库
            await self._load_exercise_library()
            
            logger.info(f"健身教练智能体 {self.agent_id} 初始化完成")
            
        except Exception as e:
            logger.error(f"健身教练智能体初始化失败: {e}")
            raise
    
    async def _agent_stop(self) -> None:
        """停止健身教练智能体"""
        if self.pose_analyzer:
            await self.pose_analyzer.cleanup()
        
        if self.user_profile_db:
            await self.user_profile_db.close()
    
    async def _register_capabilities(self) -> None:
        """注册健身教练能力"""
        capabilities = [
            AgentCapability(
                name="create_workout_plan",
                description="创建个性化训练计划",
                input_types=["user_profile", "fitness_goals"],
                output_types=["workout_plan"]
            ),
            AgentCapability(
                name="analyze_exercise_form",
                description="分析运动动作标准度",
                input_types=["video", "image"],
                output_types=["form_analysis", "feedback"]
            ),
            AgentCapability(
                name="track_progress",
                description="跟踪训练进度",
                input_types=["workout_data"],
                output_types=["progress_report", "recommendations"]
            ),
            AgentCapability(
                name="provide_exercise_guidance",
                description="提供运动指导",
                input_types=["exercise_name", "user_level"],
                output_types=["instructions", "tips", "safety_notes"]
            ),
            AgentCapability(
                name="assess_injury_risk",
                description="评估运动伤害风险",
                input_types=["exercise_data", "user_history"],
                output_types=["risk_assessment", "prevention_tips"]
            )
        ]
        
        for capability in capabilities:
            self.register_capability(capability)
    
    async def _process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """处理健身教练请求"""
        request_type = request.get("type")
        user_id = request.get("user_id")
        
        try:
            if request_type == "create_workout_plan":
                return await self._create_workout_plan(request)
            
            elif request_type == "analyze_exercise_form":
                return await self._analyze_exercise_form(request)
            
            elif request_type == "track_progress":
                return await self._track_progress(request)
            
            elif request_type == "provide_guidance":
                return await self._provide_exercise_guidance(request)
            
            elif request_type == "assess_risk":
                return await self._assess_injury_risk(request)
            
            elif request_type == "chat":
                return await self._handle_fitness_chat(request)
            
            else:
                return {
                    "success": False,
                    "error": f"不支持的请求类型: {request_type}",
                    "supported_types": [
                        "create_workout_plan", "analyze_exercise_form", 
                        "track_progress", "provide_guidance", "assess_risk", "chat"
                    ]
                }
                
        except Exception as e:
            logger.error(f"健身教练处理请求失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now()
            }
    
    async def _create_workout_plan(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """创建个性化训练计划"""
        user_id = request.get("user_id")
        goals = request.get("goals", [])
        duration_weeks = request.get("duration_weeks", 8)
        fitness_level = request.get("fitness_level", "beginner")
        available_equipment = request.get("equipment", [])
        time_per_session = request.get("time_per_session", 60)  # 分钟
        sessions_per_week = request.get("sessions_per_week", 3)
        
        try:
            # 获取用户档案
            user_profile = await self.user_profile_db.get_user_profile(user_id)
            if not user_profile:
                user_profile = await self._create_default_profile(user_id, request)
            
            # 分析用户需求
            plan_requirements = await self._analyze_plan_requirements(
                user_profile, goals, fitness_level, available_equipment
            )
            
            # 生成训练计划
            workout_plan = await self._generate_workout_plan(
                user_id, plan_requirements, duration_weeks, 
                time_per_session, sessions_per_week
            )
            
            # 保存计划
            self.workout_plans[workout_plan.plan_id] = workout_plan
            
            # 使用RAG增强计划说明
            enhanced_explanation = await self._enhance_plan_explanation(workout_plan)
            
            return {
                "success": True,
                "plan_id": workout_plan.plan_id,
                "workout_plan": {
                    "goal": workout_plan.goal,
                    "duration_weeks": workout_plan.duration_weeks,
                    "intensity_level": workout_plan.intensity_level,
                    "schedule": workout_plan.schedule,
                    "exercises": workout_plan.exercises,
                    "equipment_needed": workout_plan.equipment_needed
                },
                "explanation": enhanced_explanation,
                "tips": await self._get_plan_tips(workout_plan),
                "created_at": workout_plan.created_at
            }
            
        except Exception as e:
            logger.error(f"创建训练计划失败: {e}")
            raise
    
    async def _analyze_exercise_form(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """分析运动动作标准度"""
        exercise_name = request.get("exercise_name")
        media_data = request.get("media_data")  # 视频或图像数据
        media_type = request.get("media_type", "video")  # video, image
        user_id = request.get("user_id")
        
        try:
            # 姿态检测和分析
            if media_type == "video":
                analysis_result = await self.pose_analyzer.analyze_video(
                    media_data, exercise_name
                )
            else:
                analysis_result = await self.pose_analyzer.analyze_image(
                    media_data, exercise_name
                )
            
            # 生成详细反馈
            feedback = await self._generate_form_feedback(
                exercise_name, analysis_result
            )
            
            # 风险评估
            risk_assessment = await self._assess_form_risk(
                exercise_name, analysis_result
            )
            
            # 改进建议
            improvement_tips = await self._get_improvement_tips(
                exercise_name, analysis_result
            )
            
            exercise_analysis = ExerciseAnalysis(
                exercise_name=exercise_name,
                form_score=analysis_result.get("form_score", 0),
                rep_count=analysis_result.get("rep_count", 0),
                duration=analysis_result.get("duration", 0),
                calories_burned=analysis_result.get("calories_burned", 0),
                feedback=feedback,
                pose_keypoints=analysis_result.get("keypoints", []),
                risk_assessment=risk_assessment
            )
            
            # 记录分析历史
            if user_id:
                await self._record_exercise_analysis(user_id, exercise_analysis)
            
            return {
                "success": True,
                "analysis": {
                    "exercise_name": exercise_analysis.exercise_name,
                    "form_score": exercise_analysis.form_score,
                    "grade": self._get_form_grade(exercise_analysis.form_score),
                    "rep_count": exercise_analysis.rep_count,
                    "duration": exercise_analysis.duration,
                    "calories_burned": exercise_analysis.calories_burned,
                    "risk_level": risk_assessment
                },
                "feedback": feedback,
                "improvement_tips": improvement_tips,
                "next_steps": await self._get_next_steps(exercise_analysis),
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"动作分析失败: {e}")
            raise
    
    async def _track_progress(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """跟踪训练进度"""
        user_id = request.get("user_id")
        workout_data = request.get("workout_data")
        plan_id = request.get("plan_id")
        
        try:
            # 获取历史数据
            if user_id not in self.user_progress:
                self.user_progress[user_id] = []
            
            # 记录当前训练数据
            progress_entry = {
                "timestamp": datetime.now(),
                "plan_id": plan_id,
                "workout_data": workout_data,
                "performance_metrics": await self._calculate_performance_metrics(workout_data)
            }
            
            self.user_progress[user_id].append(progress_entry)
            
            # 分析进度趋势
            progress_analysis = await self._analyze_progress_trends(user_id)
            
            # 生成建议
            recommendations = await self._generate_progress_recommendations(
                user_id, progress_analysis
            )
            
            # 调整训练计划（如果需要）
            plan_adjustments = await self._suggest_plan_adjustments(
                user_id, plan_id, progress_analysis
            )
            
            return {
                "success": True,
                "progress_summary": {
                    "current_performance": progress_entry["performance_metrics"],
                    "trend_analysis": progress_analysis,
                    "achievement_rate": progress_analysis.get("achievement_rate", 0),
                    "consistency_score": progress_analysis.get("consistency_score", 0)
                },
                "recommendations": recommendations,
                "plan_adjustments": plan_adjustments,
                "motivational_message": await self._get_motivational_message(progress_analysis),
                "next_milestone": await self._get_next_milestone(user_id, plan_id)
            }
            
        except Exception as e:
            logger.error(f"进度跟踪失败: {e}")
            raise
    
    async def _provide_exercise_guidance(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """提供运动指导"""
        exercise_name = request.get("exercise_name")
        user_level = request.get("user_level", "beginner")
        specific_question = request.get("question")
        
        try:
            # 从知识库获取基础信息
            exercise_info = await self.knowledge_base.get_exercise_info(exercise_name)
            
            if not exercise_info:
                return {
                    "success": False,
                    "error": f"未找到运动 '{exercise_name}' 的相关信息"
                }
            
            # 根据用户水平调整指导内容
            level_specific_guidance = await self._adapt_guidance_to_level(
                exercise_info, user_level
            )
            
            # 生成详细指导
            guidance = {
                "exercise_name": exercise_name,
                "description": exercise_info.get("description"),
                "target_muscles": exercise_info.get("target_muscles", []),
                "equipment_needed": exercise_info.get("equipment", []),
                "difficulty_level": exercise_info.get("difficulty"),
                "step_by_step": level_specific_guidance.get("instructions", []),
                "form_tips": level_specific_guidance.get("form_tips", []),
                "common_mistakes": exercise_info.get("common_mistakes", []),
                "safety_notes": exercise_info.get("safety_notes", []),
                "modifications": level_specific_guidance.get("modifications", []),
                "progressions": exercise_info.get("progressions", [])
            }
            
            # 如果有具体问题，使用RAG回答
            if specific_question:
                rag_response = await self.knowledge_base.query(
                    f"{exercise_name} {specific_question}"
                )
                guidance["specific_answer"] = rag_response
            
            # 生成个性化建议
            personalized_tips = await self._get_personalized_tips(
                exercise_name, user_level
            )
            
            return {
                "success": True,
                "guidance": guidance,
                "personalized_tips": personalized_tips,
                "related_exercises": await self._get_related_exercises(exercise_name),
                "video_tutorials": exercise_info.get("video_links", []),
                "estimated_calories": await self._estimate_calories(exercise_name, user_level)
            }
            
        except Exception as e:
            logger.error(f"提供运动指导失败: {e}")
            raise
    
    async def _assess_injury_risk(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """评估运动伤害风险"""
        user_id = request.get("user_id")
        exercise_data = request.get("exercise_data")
        user_history = request.get("user_history")
        
        try:
            # 获取用户健康档案
            user_profile = await self.user_profile_db.get_user_profile(user_id)
            
            # 分析风险因素
            risk_factors = await self._analyze_risk_factors(
                user_profile, exercise_data, user_history
            )
            
            # 计算风险评分
            risk_score = await self._calculate_risk_score(risk_factors)
            
            # 生成风险评估报告
            risk_assessment = {
                "overall_risk_level": self._get_risk_level(risk_score),
                "risk_score": risk_score,
                "risk_factors": risk_factors,
                "high_risk_exercises": await self._identify_high_risk_exercises(
                    user_profile, exercise_data
                ),
                "prevention_strategies": await self._get_prevention_strategies(risk_factors),
                "recommended_modifications": await self._get_safety_modifications(
                    exercise_data, risk_factors
                )
            }
            
            # 生成个性化建议
            safety_recommendations = await self._generate_safety_recommendations(
                risk_assessment
            )
            
            return {
                "success": True,
                "risk_assessment": risk_assessment,
                "safety_recommendations": safety_recommendations,
                "monitoring_suggestions": await self._get_monitoring_suggestions(risk_score),
                "emergency_protocols": await self._get_emergency_protocols(),
                "follow_up_schedule": await self._suggest_follow_up_schedule(risk_score)
            }
            
        except Exception as e:
            logger.error(f"风险评估失败: {e}")
            raise
    
    async def _handle_fitness_chat(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """处理健身相关聊天"""
        user_message = request.get("message")
        user_id = request.get("user_id")
        context = request.get("context", {})
        
        try:
            # 使用RAG检索相关知识
            relevant_info = await self.knowledge_base.query(user_message)
            
            # 获取用户上下文
            user_context = await self._get_user_context(user_id)
            
            # 生成回复
            response = await self._generate_chat_response(
                user_message, relevant_info, user_context, context
            )
            
            # 检测是否需要特殊处理
            special_actions = await self._detect_special_actions(user_message)
            
            return {
                "success": True,
                "response": response,
                "special_actions": special_actions,
                "follow_up_questions": await self._generate_follow_up_questions(user_message),
                "related_topics": await self._get_related_topics(user_message),
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"健身聊天处理失败: {e}")
            raise
    
    # 辅助方法
    async def _load_exercise_library(self) -> None:
        """加载运动库"""
        # 这里可以从文件或数据库加载运动库
        # 示例数据
        self.exercise_library = {
            "push_up": {
                "name": "俯卧撑",
                "category": "胸部",
                "difficulty": "初级",
                "equipment": [],
                "target_muscles": ["胸大肌", "三角肌前束", "肱三头肌"],
                "calories_per_minute": 8
            },
            "squat": {
                "name": "深蹲",
                "category": "腿部",
                "difficulty": "初级",
                "equipment": [],
                "target_muscles": ["股四头肌", "臀大肌", "腘绳肌"],
                "calories_per_minute": 10
            }
            # 更多运动...
        }
    
    async def _create_default_profile(self, user_id: str, request: Dict[str, Any]) -> Dict[str, Any]:
        """创建默认用户档案"""
        profile = {
            "user_id": user_id,
            "age": request.get("age", 25),
            "gender": request.get("gender", "unknown"),
            "height": request.get("height", 170),  # cm
            "weight": request.get("weight", 70),   # kg
            "fitness_level": request.get("fitness_level", "beginner"),
            "health_conditions": request.get("health_conditions", []),
            "injuries": request.get("injuries", []),
            "created_at": datetime.now()
        }
        
        await self.user_profile_db.create_user_profile(profile)
        return profile
    
    def _get_form_grade(self, score: float) -> str:
        """根据评分获取等级"""
        if score >= 90:
            return "优秀"
        elif score >= 80:
            return "良好"
        elif score >= 70:
            return "及格"
        else:
            return "需要改进"
    
    def _get_risk_level(self, risk_score: float) -> str:
        """根据风险评分获取风险等级"""
        if risk_score >= 0.8:
            return "高风险"
        elif risk_score >= 0.5:
            return "中等风险"
        else:
            return "低风险"
    
    # 更多辅助方法...
    async def _analyze_plan_requirements(self, user_profile, goals, fitness_level, equipment):
        """分析计划需求"""
        # 实现计划需求分析逻辑
        pass
    
    async def _generate_workout_plan(self, user_id, requirements, duration, time_per_session, sessions_per_week):
        """生成训练计划"""
        # 实现训练计划生成逻辑
        pass
    
    # ... 其他辅助方法的实现