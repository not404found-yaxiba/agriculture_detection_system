def __init__(self, config_path):
    self.config = self.load_config(config_path)
    # 保存配置文件的绝对路径
    self.config['_config_file'] = os.path.abspath(config_path)
    self.model = None
    self.classes = self.config.get('classes', {})
    self.colors = self.config.get('colors', {})