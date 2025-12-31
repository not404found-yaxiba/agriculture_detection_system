from .base_detector import BaseDetector
import torch

class AppleDetector(BaseDetector):
    """苹果检测器 - 使用YOLOv5"""
    
    def load_model(self):
        import os
        import torch
        
        # 获取模型路径（配置文件中的相对路径）
        model_path = self.config['model_path']  # 例如: "models/apple/best.pt"
        
        print(f" 加载苹果检测模型: {model_path}")
        
        # 关键：基于配置文件目录解析相对路径
        config_dir = os.path.dirname(os.path.abspath(self.config['_config_file']))
        
        if not os.path.isabs(model_path):
            # 先尝试相对于配置文件目录
            model_abs_path = os.path.join(config_dir, model_path)
            
            # 如果文件不存在，尝试相对于项目根目录（configs的上一级）
            if not os.path.exists(model_abs_path):
                project_root = os.path.dirname(config_dir)  # configs的上二级是项目根目录
                model_abs_path = os.path.join(project_root, model_path)
                
                if not os.path.exists(model_abs_path):
                    raise FileNotFoundError(f" 找不到模型文件: {model_path}\n"
                                          f"尝试路径1: {os.path.join(config_dir, model_path)}\n"
                                          f"尝试路径2: {model_abs_path}")
            
            model_path = model_abs_path
        
        print(f"   解析后路径: {model_path}")
        
        # 加载模型
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', 
                               path=model_path, force_reload=False)
    
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(device)
        # 使用配置文件中的置信度阈值
        self.model.conf = self.config.get('confidence_threshold', 0.5)
    
        print(" 苹果检测模型加载成功")
        return True
    
    def detect_image(self, image_path):
        """检测单张图片"""
        import cv2
        import pandas as pd
        
        # 读取图片
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图片: {image_path}")
        
        # 推理
        results = self.model(img)
        detections_df = results.pandas().xyxy[0]
        
        # 转换为标准格式
        detections = []
        for _, row in detections_df.iterrows():
            detection = {
                'bbox': [int(row['xmin']), int(row['ymin']), 
                         int(row['xmax']), int(row['ymax'])],
                'confidence': row['confidence'],
                'class_id': int(row['class'])
            }
            detections.append(detection)
        
        # 绘制结果
        output_img = self.draw_detections(img, detections)
        
        return output_img, detections
