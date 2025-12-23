import cv2
import numpy as np
import yaml
import os
from abc import ABC, abstractmethod

class BaseDetector(ABC):
    def __init__(self, config_path):
        # 🔧 确保导入os并保存配置文件路径
        self.config = self.load_config(config_path)
        self.config['_config_file'] = os.path.abspath(config_path)  # 添加这行
        self.model = None
        self.classes = self.config.get('classes', {})
        self.colors = self.config.get('colors', {})
    
    def load_config(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    @abstractmethod
    def load_model(self):
        pass
    
    @abstractmethod
    def detect_image(self, image_path):
        pass
    
    def draw_detections(self, image, detections):
        output_img = image.copy()
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            class_id = det['class_id']
            class_name = self.classes.get(class_id, f'class_{class_id}')
            color = self.colors.get(class_name, (0, 255, 0))
            
            cv2.rectangle(output_img, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name} {conf:.2f}"
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(output_img, 
                         (x1, y1 - label_height - baseline - 5),
                         (x1 + label_width, y1),
                         color, -1)
            cv2.putText(output_img, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        return output_img