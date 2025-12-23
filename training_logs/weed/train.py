# train.py - 专用于杂草检测模型训练
from ultralytics import YOLO
import os

print("当前工作目录:", os.getcwd())

def main():
    # 1. 加载预训练模型
    print("[1/3] 正在加载预训练模型 yolov8n.pt...")
    model = YOLO('yolov8n.pt')

    # 2. 开始训练（关键：data.yaml就在当前目录，所以直接用‘data.yaml’）
    print("[2/3] 开始训练，请耐心等待...")
    results = model.train(
        data='data.yaml',
        epochs=50,       # 大幅增加轮次
        batch=8,
        imgsz=640,
        device='cpu',
        lr0=0.001,        # 显著降低学习率
        patience=20,      # 新增：早停耐心值
        project='weed_training_result',  # 训练输出也会生成在这个目录下
        name='exp1',
        exist_ok=True,
        save_period=10,
        augment=True
    )
    print(f"[3/3] 训练完成！最佳模型保存在: weed_training_result/exp1/weights/best.pt")

if __name__ == '__main__':
    main()