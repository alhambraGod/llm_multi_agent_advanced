"""
智能体协作示例

展示多个智能体如何协作为用户提供综合性的健身指导
"""

import asyncio
from datetime import datetime
from src.core.system import MultiAgentSystem

async def comprehensive_fitness_consultation_example():
    """综合健身咨询示例"""
    
    # 初始化多智能体系统
    system = MultiAgentSystem()
    await system.start()
    
    # 用户请求：制定综合健身计划
    user_request = {
        "user_id": "user_123",
        "type": "comprehensive_plan",
        "goals": ["减脂", "增肌", "改善体能"],
        "current_stats": {
            "age": 28,
            "weight": 75,
            "height": 175,
            "body_fat": 18,
            "fitness_level": "intermediate"
        },
        "preferences": {
            "workout_frequency": 4,
            "session_duration": 60,
            "dietary_restrictions": ["无乳糖"],
            "equipment_access": ["健身房"]
        }
    }
    
    # 1. 健身教练制定训练计划
    fitness_plan = await system.execute_task({
        "task_id": "create_fitness_plan",
        "type": "create_workout_plan",
        "agent_id": "fitness_coach",
        "user_id": user_request["user_id"],
        "goals": user_request["goals"],
        "duration_weeks": 12,
        "fitness_level": user_request["current_stats"]["fitness_level"],
        "sessions_per_week": user_request["preferences"]["workout_frequency"],
        "time_per_session": user_request["preferences"]["session_duration"]
    })
    
    # 2. 营养师制定饮食计划
    nutrition_plan = await system.execute_task({
        "task_id": "create_nutrition_plan",
        "type": "create_nutrition_plan",
        "agent_id": "nutrition_expert",
        "user_id": user_request["user_id"],
        "goals": user_request["goals"],
        "dietary_restrictions": user_request["preferences"]["dietary_restrictions"],
        "activity_level": "high",  # 基于训练计划
        "duration_days": 84  # 12周
    })
    
    # 3. 心理健康顾问评估心理状态
    mental_assessment = await system.execute_task({
        "task_id": "assess_mental_health",
        "type": "assess_mental_health",
        "agent_id": "mental_health",
        "user_id": user_request["user_id"],
        "questionnaire_data": {
            "stress_level": 6,  # 1-10
            "motivation_level": 8,
            "sleep_quality": 7,
            "work_life_balance": 5
        }
    })
    
    # 4. 数据分析师生成基线分析
    baseline_analysis = await system.execute_task({
        "task_id": "baseline_analysis",
        "type": "generate_insights",
        "agent_id": "data_analyst",
        "user_id": user_request["user_id"],
        "user_data": user_request["current_stats"]
    })
    
    # 5. 整合所有建议
    comprehensive_plan = {
        "user_id": user_request["user_id"],
        "created_at": datetime.now(),
        "fitness_plan": fitness_plan,
        "nutrition_plan": nutrition_plan,
        "mental_health_assessment": mental_assessment,
        "baseline_analysis": baseline_analysis,
        "integrated_recommendations": [
            "建议从中等强度训练开始，逐步增加难度",
            "注意蛋白质摄入，支持肌肉增长",
            "保持良好的睡眠质量，有助于恢复",
            "定期监测进度，及时调整计划"
        ]
    }
    
    print("=== 综合健身计划 ===")
    print(f"用户ID: {comprehensive_plan['user_id']}")
    print(f"创建时间: {comprehensive_plan['created_at']}")
    print("\n训练计划概要:")
    print(f"- 目标: {fitness_plan.get('workout_plan', {}).get('goal', 'N/A')}")
    print(f"- 持续时间: {fitness_plan.get('workout_plan', {}).get('duration_weeks', 'N/A')} 周")
    print(f"- 强度等级: {fitness_plan.get('workout_plan', {}).get('intensity_level', 'N/A')}")
    
    print("\n营养计划概要:")
    print(f"- 目标: {nutrition_plan.get('nutrition_plan', {}).get('goal', 'N/A')}")
    print(f"- 每日热量: {nutrition_plan.get('nutrition_plan', {}).get('daily_calories', 'N/A')} 卡")
    
    print("\n心理健康评估:")
    print(f"- 压力水平: {mental_assessment.get('assessment', {}).get('stress_level', 'N/A')}")
    print(f"- 动机水平: {mental_assessment.get('assessment', {}).get('motivation_level', 'N/A')}")
    
    await system.stop()
    return comprehensive_plan

if __name__ == "__main__":
    asyncio.run(comprehensive_fitness_consultation_example())