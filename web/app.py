import os
import sys
import importlib.util
from flask import Flask, request, render_template, send_from_directory, jsonify
from werkzeug.utils import secure_filename
import cv2
import json

# ========== 路径配置 ==========
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

# 关键：将src目录作为包来处理
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SRC_PATH)

print(f"项目根目录: {PROJECT_ROOT}")

# ========== 动态导入检测器 ==========
print("正在导入检测器...")

try:
    # 1. 先加载base_detector
    base_spec = importlib.util.spec_from_file_location(
        "src.base_detector", 
        os.path.join(SRC_PATH, "base_detector.py")
    )
    base_module = importlib.util.module_from_spec(base_spec)
    sys.modules["src.base_detector"] = base_module
    sys.modules["base_detector"] = base_module
    base_spec.loader.exec_module(base_module)

    # 2. 加载apple_detector
    apple_spec = importlib.util.spec_from_file_location(
        "src.apple_detector",
        os.path.join(SRC_PATH, "apple_detector.py")
    )
    apple_module = importlib.util.module_from_spec(apple_spec)
    sys.modules["src.apple_detector"] = apple_module
    apple_module.__dict__["__package__"] = "src"
    apple_spec.loader.exec_module(apple_module)

    # 3. 加载weed_detector
    weed_spec = importlib.util.spec_from_file_location(
        "src.weed_detector",
        os.path.join(SRC_PATH, "weed_detector.py")
    )
    weed_module = importlib.util.module_from_spec(weed_spec)
    sys.modules["src.weed_detector"] = weed_module
    weed_module.__dict__["__package__"] = "src"
    weed_spec.loader.exec_module(weed_module)

    # 获取类
    AppleDetector = getattr(apple_module, "AppleDetector")
    WeedDetector = getattr(weed_module, "WeedDetector")

    print(f" AppleDetector类: {AppleDetector}")
    print(f" WeedDetector类: {WeedDetector}")

except Exception as e:
    print(f" 导入失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ========== Flask应用配置 ==========
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ========== 全局检测器实例 ==========
apple_detector = None
weed_detector = None
current_image_path = None  # 存储当前处理的图片路径

def init_detectors():
    global apple_detector, weed_detector
    
    try:
        original_cwd = os.getcwd()
        os.chdir(PROJECT_ROOT)
        print(f" 切换到项目根目录: {PROJECT_ROOT}")
        
        apple_config_path = 'configs/apple_config.yaml'
        weed_config_path = 'configs/weed_config.yaml'
        
        print(f"📄 苹果配置文件: {apple_config_path}")
        print(f"📄 杂草配置文件: {weed_config_path}")
        
        if not os.path.exists(apple_config_path):
            print(f" 配置文件不存在: {apple_config_path}")
            os.chdir(original_cwd)
            return False
        
        apple_detector = AppleDetector(apple_config_path)
        if not apple_detector.load_model():
            os.chdir(original_cwd)
            return False
        
        weed_detector = WeedDetector(weed_config_path)
        if not weed_detector.load_model():
            os.chdir(original_cwd)
            return False
        
        os.chdir(original_cwd)
        print(" 检测器初始化成功")
        return True
        
    except Exception as e:
        print(f" 检测器初始化失败: {e}")
        import traceback
        traceback.print_exc()
        if 'original_cwd' in locals():
            os.chdir(original_cwd)
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    global current_image_path
    
    if 'file' not in request.files:
        return "没有上传文件", 400
    
    file = request.files['file']
    mode = request.form.get('mode', 'apple')
    
    if file.filename == '':
        return "没有选择文件", 400
    
    if not allowed_file(file.filename):
        return "只支持PNG、JPG、JPEG、BMP格式", 400
    
    global apple_detector, weed_detector
    if apple_detector is None or weed_detector is None:
        if not init_detectors():
            return "检测器初始化失败", 500
    
    try:
        filename = secure_filename(file.filename)
        temp_dir = 'temp_uploads'
        os.makedirs(temp_dir, exist_ok=True)
        
        import uuid
        unique_id = str(uuid.uuid4())[:8]
        base_name, ext = os.path.splitext(filename)
        unique_filename = f"{base_name}_{unique_id}{ext}"
        upload_path = os.path.join(temp_dir, unique_filename)
        
        file.save(upload_path)
        print(f" 文件保存到: {upload_path}")
        
        # 保存当前图片路径供后续使用
        current_image_path = upload_path
        
        result_dir = 'static/results'
        os.makedirs(result_dir, exist_ok=True)
        
        result_data = {'result': True}
        
        if mode in ['apple', 'both']:
            apple_img, apple_dets = apple_detector.detect_image(upload_path)
            if apple_img is not None:
                apple_result_path = os.path.join(result_dir, f'apple_{unique_filename}')
                cv2.imwrite(apple_result_path, apple_img)
                result_data['apple_img'] = f'/static/results/apple_{unique_filename}'
                result_data['apple_count'] = len(apple_dets) if apple_dets else 0
        
        if mode in ['weed', 'both']:
            weed_img, weed_dets = weed_detector.detect_image(upload_path)
            if weed_img is not None:
                weed_result_path = os.path.join(result_dir, f'weed_{unique_filename}')
                cv2.imwrite(weed_result_path, weed_img)
                result_data['weed_img'] = f'/static/results/weed_{unique_filename}'
                result_data['weed_count'] = len(weed_dets) if weed_dets else 0
        
        # 保存原始图片路径到结果数据中
        result_data['original_path'] = upload_path
        
        return render_template('index.html', **result_data)
        
    except Exception as e:
        print(f" 检测失败: {e}")
        import traceback
        traceback.print_exc()
        return f"检测失败: {str(e)}", 500

@app.route('/update_confidence', methods=['POST'])
def update_confidence():
    """更新置信度并重新检测"""
    try:
        data = request.json
        new_confidence = float(data.get('confidence', 0.5))
        mode = data.get('mode', 'both')
        image_path = data.get('image_path')
        
        print(f"收到置信度更新请求: {new_confidence}, 模式: {mode}, 图片: {image_path}")
        
        if not image_path or not os.path.exists(image_path):
            return jsonify({'success': False, 'error': '图片文件不存在'})
        
        # 更新检测器的置信度
        global apple_detector, weed_detector
        
        if mode in ['apple', 'both'] and apple_detector:
            apple_detector.model.conf = new_confidence
            print(f"苹果检测器置信度已更新: {new_confidence}")
            
        if mode in ['weed', 'both'] and weed_detector:
            weed_detector.model.overrides['conf'] = new_confidence
            print(f"杂草检测器置信度已更新: {new_confidence}")
        
        # 重新检测
        result_data = {'success': True, 'confidence': new_confidence}
        
        if mode in ['apple', 'both']:
            apple_img, apple_dets = apple_detector.detect_image(image_path)
            if apple_img is not None:
                import uuid
                unique_id = str(uuid.uuid4())[:8]
                apple_result_name = f'apple_conf_{new_confidence}_{unique_id}.jpg'
                apple_result_path = os.path.join('static', 'results', apple_result_name)
                cv2.imwrite(apple_result_path, apple_img)
                
                result_data['apple_img'] = f'/static/results/{apple_result_name}'
                result_data['apple_count'] = len(apple_dets) if apple_dets else 0
        
        if mode in ['weed', 'both']:
            weed_img, weed_dets = weed_detector.detect_image(image_path)
            if weed_img is not None:
                import uuid
                unique_id = str(uuid.uuid4())[:8]
                weed_result_name = f'weed_conf_{new_confidence}_{unique_id}.jpg'
                weed_result_path = os.path.join('static', 'results', weed_result_name)
                cv2.imwrite(weed_result_path, weed_img)
                
                result_data['weed_img'] = f'/static/results/{weed_result_name}'
                result_data['weed_count'] = len(weed_dets) if weed_dets else 0
        
        return jsonify(result_data)
        
    except Exception as e:
        print(f"更新置信度时出错: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    print("=" * 50)
    print(" 农业检测Web服务 - 支持置信度实时更新")
    print("=" * 50)
    
    os.makedirs('temp_uploads', exist_ok=True)
    os.makedirs('static/results', exist_ok=True)
    
    if init_detectors():
        print(" 系统准备就绪")
    else:
        print("  检测器初始化失败")
    
    print(f"访问地址: http://127.0.0.1:5000")
    print("=" * 50)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
