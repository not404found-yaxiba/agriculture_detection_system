from .base_detector import BaseDetector

class WeedDetector(BaseDetector):
    """杂草检测器 - 使用YOLOv8"""
    
    def load_model(self):
        import os
        from ultralytics import YOLO
        
        model_path = self.config['model_path']
        
        print(f" 加载杂草检测模型: {model_path}")
        
        # 同样的相对路径解析逻辑
        config_dir = os.path.dirname(os.path.abspath(self.config['_config_file']))
        
        if not os.path.isabs(model_path):
            model_abs_path = os.path.join(config_dir, model_path)
            
            if not os.path.exists(model_abs_path):
                project_root = os.path.dirname(config_dir)
                model_abs_path = os.path.join(project_root, model_path)
                
                if not os.path.exists(model_abs_path):
                    raise FileNotFoundError(f" 找不到模型文件: {model_path}\n"
                                          f"尝试路径1: {os.path.join(config_dir, model_path)}\n"
                                          f"尝试路径2: {model_abs_path}")
            
            model_path = model_abs_path
        
        print(f"   解析后路径: {model_path}")
        
        # 加载模型
        self.model = YOLO(model_path)
        # 设置置信度阈值
        self.model.overrides['conf'] = self.config.get('confidence_threshold', 0.5)
    
        print(" 杂草检测模型加载成功")
        return True
    
    def detect_image(self, image_path):
        """检测单张图片"""
        import cv2
        
        # 读取图片
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图片: {image_path}")
        
        # YOLOv8推理
        results = self.model.predict(img, conf=self.config.get('confidence_threshold', 0.3))
        
        # 解析结果
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                
                detection = {
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': conf,
                    'class_id': cls
                }
                detections.append(detection)
        
        # 绘制结果
        output_img = self.draw_detections(img, detections)
        
        return output_img, detections
