"""
多模态处理器 - 统一处理文本、图像、音频、视频等多种模态数据
"""

import logging
from typing import List, Dict, Any, Optional, Union, Tuple
import asyncio
from datetime import datetime
import base64
import io

# 图像处理
from PIL import Image
import cv2
import numpy as np

# 音频处理
import librosa
import soundfile as sf

# 视频处理
from moviepy.editor import VideoFileClip

# AI模型
import torch
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    Wav2Vec2Processor, Wav2Vec2ForCTC,
    CLIPProcessor, CLIPModel
)

# 本地模块
from .image_processor import ImageProcessor
from .audio_processor import AudioProcessor
from .video_processor import VideoProcessor
from .text_processor import TextProcessor

class MultimodalProcessor:
    """
    企业级多模态处理器
    统一处理和融合多种模态的数据
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化多模态处理器
        
        Args:
            config: 配置参数
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 各模态处理器
        self.image_processor = ImageProcessor(config.get('image', {}))
        self.audio_processor = AudioProcessor(config.get('audio', {}))
        self.video_processor = VideoProcessor(config.get('video', {}))
        self.text_processor = TextProcessor(config.get('text', {}))
        
        # 多模态模型
        self.clip_model = None
        self.clip_processor = None
        self.blip_model = None
        self.blip_processor = None
        
        # 支持的模态类型
        self.supported_modalities = {
            'text': self._process_text,
            'image': self._process_image,
            'audio': self._process_audio,
            'video': self._process_video
        }
        
        # 性能监控
        self.metrics = {
            'total_processed': 0,
            'by_modality': {
                'text': 0,
                'image': 0,
                'audio': 0,
                'video': 0
            },
            'avg_processing_time': 0.0,
            'successful_fusions': 0
        }
    
    async def initialize(self):
        """初始化多模态处理器"""
        try:
            # 初始化各模态处理器
            await self.image_processor.initialize()
            await self.audio_processor.initialize()
            await self.video_processor.initialize()
            await self.text_processor.initialize()
            
            # 加载多模态模型
            await self._load_multimodal_models()
            
            self.logger.info(f"多模态处理器初始化完成 - 设备: {self.device}")
            
        except Exception as e:
            self.logger.error(f"多模态处理器初始化失败: {e}")
            raise
    
    async def _load_multimodal_models(self):
        """加载多模态模型"""
        try:
            # 加载CLIP模型（图文对齐）
            if self.config.get('enable_clip', True):
                self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                self.clip_model.to(self.device)
                self.logger.info("CLIP模型加载完成")
            
            # 加载BLIP模型（图像描述）
            if self.config.get('enable_blip', True):
                self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
                self.blip_model.to(self.device)
                self.logger.info("BLIP模型加载完成")
                
        except Exception as e:
            self.logger.error(f"多模态模型加载失败: {e}")
            raise
    
    async def process_multimodal_input(self, 
                                     inputs: List[Dict[str, Any]], 
                                     fusion_strategy: str = 'concatenate') -> Dict[str, Any]:
        """
        处理多模态输入
        
        Args:
            inputs: 多模态输入列表，每个元素包含 {'type': 'text/image/audio/video', 'data': ...}
            fusion_strategy: 融合策略
            
        Returns:
            处理结果
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            # 分别处理各模态
            processed_results = []
            
            for input_item in inputs:
                modality_type = input_item.get('type')
                data = input_item.get('data')
                
                if modality_type not in self.supported_modalities:
                    self.logger.warning(f"不支持的模态类型: {modality_type}")
                    continue
                
                # 处理单个模态
                processor_func = self.supported_modalities[modality_type]
                result = await processor_func(data, input_item.get('metadata', {}))
                
                if result:
                    result['modality'] = modality_type
                    processed_results.append(result)
                    self.metrics['by_modality'][modality_type] += 1
            
            # 融合多模态结果
            fused_result = await self._fuse_multimodal_results(
                processed_results, 
                fusion_strategy
            )
            
            # 更新指标
            processing_time = asyncio.get_event_loop().time() - start_time
            self._update_metrics(processing_time, len(inputs))
            
            return {
                'fused_result': fused_result,
                'individual_results': processed_results,
                'processing_time': processing_time,
                'processed_modalities': [r['modality'] for r in processed_results]
            }
            
        except Exception as e:
            self.logger.error(f"多模态处理失败: {e}")
            return {
                'fused_result': None,
                'individual_results': [],
                'error': str(e)
            }
    
    async def _process_text(self, data: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """处理文本数据"""
        try:
            result = await self.text_processor.process(data, metadata)
            return {
                'content': data,
                'processed_content': result.get('processed_text', data),
                'embeddings': result.get('embeddings'),
                'features': result.get('features', {}),
                'metadata': metadata
            }
        except Exception as e:
            self.logger.error(f"文本处理失败: {e}")
            return None
    
    async def _process_image(self, data: Union[str, bytes, Image.Image], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """处理图像数据"""
        try:
            # 统一图像格式
            if isinstance(data, str):
                # 文件路径或base64
                if data.startswith('data:image'):
                    # base64图像
                    image_data = base64.b64decode(data.split(',')[1])
                    image = Image.open(io.BytesIO(image_data))
                else:
                    # 文件路径
                    image = Image.open(data)
            elif isinstance(data, bytes):
                image = Image.open(io.BytesIO(data))
            else:
                image = data
            
            # 基础图像处理
            basic_result = await self.image_processor.process(image, metadata)
            
            # 生成图像描述（如果启用BLIP）
            description = ""
            if self.blip_model and self.blip_processor:
                description = await self._generate_image_description(image)
            
            # 提取图像特征（如果启用CLIP）
            image_features = None
            if self.clip_model and self.clip_processor:
                image_features = await self._extract_image_features(image)
            
            return {
                'content': description,
                'image_data': basic_result.get('processed_image'),
                'features': {
                    **basic_result.get('features', {}),
                    'clip_features': image_features
                },
                'description': description,
                'metadata': {
                    **metadata,
                    'image_size': image.size,
                    'image_mode': image.mode
                }
            }
            
        except Exception as e:
            self.logger.error(f"图像处理失败: {e}")
            return None
    
    async def _process_audio(self, data: Union[str, bytes, np.ndarray], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """处理音频数据"""
        try:
            # 基础音频处理
            basic_result = await self.audio_processor.process(data, metadata)
            
            # 语音转文字（如果有相应模型）
            transcription = basic_result.get('transcription', '')
            
            return {
                'content': transcription,
                'audio_features': basic_result.get('features', {}),
                'transcription': transcription,
                'metadata': {
                    **metadata,
                    'duration': basic_result.get('duration', 0),
                    'sample_rate': basic_result.get('sample_rate', 0)
                }
            }
            
        except Exception as e:
            self.logger.error(f"音频处理失败: {e}")
            return None
    
    async def _process_video(self, data: Union[str, bytes], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """处理视频数据"""
        try:
            # 基础视频处理
            basic_result = await self.video_processor.process(data, metadata)
            
            # 提取关键帧
            key_frames = basic_result.get('key_frames', [])
            
            # 处理关键帧（生成描述）
            frame_descriptions = []
            if key_frames and self.blip_model:
                for frame in key_frames[:5]:  # 限制处理的帧数
                    desc = await self._generate_image_description(frame)
                    frame_descriptions.append(desc)
            
            # 音频转录
            audio_transcription = basic_result.get('audio_transcription', '')
            
            # 综合内容描述
            content_parts = []
            if frame_descriptions:
                content_parts.append(f"视觉内容: {'; '.join(frame_descriptions)}")
            if audio_transcription:
                content_parts.append(f"音频内容: {audio_transcription}")
            
            return {
                'content': ' | '.join(content_parts),
                'video_features': basic_result.get('features', {}),
                'frame_descriptions': frame_descriptions,
                'audio_transcription': audio_transcription,
                'metadata': {
                    **metadata,
                    'duration': basic_result.get('duration', 0),
                    'fps': basic_result.get('fps', 0),
                    'resolution': basic_result.get('resolution', (0, 0))
                }
            }
            
        except Exception as e:
            self.logger.error(f"视频处理失败: {e}")
            return None
    
    async def _generate_image_description(self, image: Image.Image) -> str:
        """生成图像描述"""
        try:
            inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                out = self.blip_model.generate(**inputs, max_length=50)
                description = self.blip_processor.decode(out[0], skip_special_tokens=True)
            
            return description
            
        except Exception as e:
            self.logger.error(f"图像描述生成失败: {e}")
            return ""
    
    async def _extract_image_features(self, image: Image.Image) -> Optional[torch.Tensor]:
        """提取图像特征"""
        try:
            inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
            
            return image_features.cpu().numpy()
            
        except Exception as e:
            self.logger.error(f"图像特征提取失败: {e}")
            return None
    
    async def _fuse_multimodal_results(self, 
                                     results: List[Dict[str, Any]], 
                                     strategy: str) -> Dict[str, Any]:
        """
        融合多模态结果
        
        Args:
            results: 各模态处理结果
            strategy: 融合策略
            
        Returns:
            融合后的结果
        """
        if not results:
            return {}
        
        try:
            if strategy == 'concatenate':
                return await self._concatenate_fusion(results)
            elif strategy == 'weighted':
                return await self._weighted_fusion(results)
            elif strategy == 'attention':
                return await self._attention_fusion(results)
            else:
                return await self._concatenate_fusion(results)
                
        except Exception as e:
            self.logger.error(f"多模态融合失败: {e}")
            return {}
    
    async def _concatenate_fusion(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """拼接融合策略"""
        content_parts = []
        all_features = {}
        all_metadata = {}
        
        for result in results:
            # 拼接内容
            if result.get('content'):
                modality = result.get('modality', 'unknown')
                content_parts.append(f"[{modality.upper()}] {result['content']}")
            
            # 合并特征
            features = result.get('features', {})
            for key, value in features.items():
                all_features[f"{result.get('modality', 'unknown')}_{key}"] = value
            
            # 合并元数据
            metadata = result.get('metadata', {})
            for key, value in metadata.items():
                all_metadata[f"{result.get('modality', 'unknown')}_{key}"] = value
        
        return {
            'fused_content': ' | '.join(content_parts),
            'fused_features': all_features,
            'fused_metadata': all_metadata,
            'fusion_strategy': 'concatenate'
        }
    
    async def _weighted_fusion(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """加权融合策略"""
        # 模态权重
        modality_weights = {
            'text': 0.4,
            'image': 0.3,
            'audio': 0.2,
            'video': 0.1
        }
        
        weighted_content = []
        for result in results:
            modality = result.get('modality', 'text')
            weight = modality_weights.get(modality, 0.1)
            content = result.get('content', '')
            
            if content:
                weighted_content.append({
                    'content': content,
                    'weight': weight,
                    'modality': modality
                })
        
        # 按权重排序
        weighted_content.sort(key=lambda x: x['weight'], reverse=True)
        
        fused_content = ' | '.join([item['content'] for item in weighted_content])
        
        return {
            'fused_content': fused_content,
            'fusion_strategy': 'weighted',
            'content_weights': {item['modality']: item['weight'] for item in weighted_content}
        }
    
    async def _attention_fusion(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """注意力融合策略（简化版）"""
        # 简化的注意力机制：基于内容长度和模态重要性
        attention_scores = []
        
        for result in results:
            content = result.get('content', '')
            modality = result.get('modality', 'text')
            
            # 基于内容长度的分数
            length_score = min(len(content) / 100, 1.0) if content else 0
            
            # 基于模态类型的分数
            modality_score = {
                'text': 0.8,
                'image': 0.7,
                'audio': 0.6,
                'video': 0.5
            }.get(modality, 0.3)
            
            attention_score = (length_score + modality_score) / 2
            attention_scores.append(attention_score)
        
        # 归一化注意力分数
        total_score = sum(attention_scores)
        if total_score > 0:
            attention_scores = [score / total_score for score in attention_scores]
        
        # 基于注意力分数融合内容
        fused_parts = []
        for i, result in enumerate(results):
            content = result.get('content', '')
            if content and attention_scores[i] > 0.1:  # 过滤低分内容
                fused_parts.append(f"({attention_scores[i]:.2f}) {content}")
        
        return {
            'fused_content': ' | '.join(fused_parts),
            'fusion_strategy': 'attention',
            'attention_scores': attention_scores
        }
    
    def _update_metrics(self, processing_time: float, input_count: int):
        """更新性能指标"""
        self.metrics['total_processed'] += input_count
        
        # 更新平均处理时间
        if self.metrics['total_processed'] == input_count:
            self.metrics['avg_processing_time'] = processing_time
        else:
            total_time = self.metrics['avg_processing_time'] * (self.metrics['total_processed'] - input_count)
            self.metrics['avg_processing_time'] = (total_time + processing_time) / self.metrics['total_processed']
    
    async def health_check(self) -> bool:
        """健康检查"""
        try:
            # 检查各模态处理器
            checks = [
                await self.image_processor.health_check(),
                await self.audio_processor.health_check(),
                await self.video_processor.health_check(),
                await self.text_processor.health_check()
            ]
            
            return all(checks)
            
        except Exception as e:
            self.logger.error(f"多模态处理器健康检查失败: {e}")
            return False
    
    async def get_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        return {
            **self.metrics,
            'device': str(self.device),
            'models_loaded': {
                'clip': self.clip_model is not None,
                'blip': self.blip_model is not None
            },
            'timestamp': datetime.now().isoformat()
        }
    
    async def cleanup(self):
        """清理资源"""
        try:
            # 清理各模态处理器
            await self.image_processor.cleanup()
            await self.audio_processor.cleanup()
            await self.video_processor.cleanup()
            await self.text_processor.cleanup()
            
            # 清理模型
            if self.clip_model:
                del self.clip_model
                self.clip_model = None
            
            if self.blip_model:
                del self.blip_model
                self.blip_model = None
            
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.logger.info("多模态处理器资源清理完成")
            
        except Exception as e:
            self.logger.error(f"多模态处理器资源清理失败: {e}")