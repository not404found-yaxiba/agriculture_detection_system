# D:\apple_detection_roboflow\apple_detection_project\train.py
import os
import subprocess
import sys

def setup_project():
    """设置项目环境"""
    print("=" * 50)
    print("苹果检测项目 - 训练开始")
    print("=" * 50)
    
    # 1. 检查数据集
    data_path = "data.yaml"
    if not os.path.exists(data_path):
        print(f"错误: 找不到 {data_path}")
        print("请确保 data.yaml 文件存在")
        return False
    
    # 2. 检查YOLOv5
    if not os.path.exists("yolov5"):
        print("正在克隆YOLOv5...")
        result = subprocess.run(
            ["git", "clone", "https://github.com/ultralytics/yolov5.git"],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print("克隆失败，尝试手动下载")
            print("请访问: https://github.com/ultralytics/yolov5")
            return False
        print("YOLOv5克隆成功")
    
    # 3. 安装依赖
    print("\n安装依赖...")
    requirements_file = "yolov5/requirements.txt"
    if os.path.exists(requirements_file):
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", requirements_file],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print("依赖安装可能有问题，但继续尝试...")
            print(result.stderr[:500])  # 显示前500字符错误
    
    return True

def train_model():
    """开始训练模型"""
    
    # 训练参数（你可以调整这些）
    params = {
        "data": "data.yaml",          # 数据集配置
        "epochs": 60,                # 训练轮数
        "batch-size": 16,             # 批次大小（根据显存调整）
        "img-size": 640,              # 图片大小
        "weights": "yolov5s.pt",      # 预训练权重
        "project": "runs/train",      # 保存路径
        "name": "apple_exp",          # 实验名称
        "exist-ok": True,             # 覆盖已有实验
    }
    
    # 检查GPU
    import torch
    if torch.cuda.is_available():
        print(f"检测到GPU: {torch.cuda.get_device_name(0)}")
        device = "0"
    else:
        print("使用CPU训练（会比较慢）")
        device = "cpu"
    
    # 构建训练命令
    cmd = [
        sys.executable, "yolov5/train.py",
        f"--data={params['data']}",
        f"--epochs={params['epochs']}",
        f"--batch-size={params['batch-size']}",
        f"--imgsz={params['img-size']}",
        f"--weights={params['weights']}",
        f"--project={params['project']}",
        f"--name={params['name']}",
        f"--device={device}",
    ]
    
    if params["exist-ok"]:
        cmd.append("--exist-ok")
    
    print("\n训练配置:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    print(f"\n训练命令: {' '.join(cmd)}")
    print("\n开始训练...（这可能需要一些时间）")
    print("=" * 50)
    
    # 执行训练
    try:
        subprocess.run(cmd, check=True)
        
        # 检查结果
        best_model = f"{params['project']}/{params['name']}/weights/best.pt"
        if os.path.exists(best_model):
            print(f"\n✅ 训练完成！")
            print(f"最佳模型: {best_model}")
            
            # 复制到当前目录
            import shutil
            shutil.copy(best_model, "best.pt")
            print("已复制 best.pt 到当前目录")
        else:
            print("⚠️  训练完成，但未找到最佳模型文件")
            
    except subprocess.CalledProcessError as e:
        print(f"❌ 训练失败: {e}")
        return False
    
    return True

def main():
    """主函数"""
    try:
        # 1. 设置项目
        if not setup_project():
            return
        
        # 2. 开始训练
        train_model()
        
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()