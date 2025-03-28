import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import torch
model_path = r'D:/碩士班/程式測驗/ultralytics-main/ultralytics/cfg/models/11/yolo11.yaml'   #僅載入模型
data_path = r'D:/碩士班/程式測驗/ultralytics-main/ultralytics/cfg/datasets/african-wildlife.yaml'

if __name__ == '__main__':
    model = YOLO(model_path)
    model.train(
        data = data_path,
        cache = False,
        imgsz = 640,
        epochs = 1,
        single_cls = False,  #單類別檢測
        batch = 4,
        workers = 0,
        device="cuda" if torch.cuda.is_available() else "cpu",
        optimizer = 'Adam',
        amp = True,         #启用自动混合精度(AMP) 训练，可减少内存使用量并加快训练速度，同时将对精度的影响降至最低。
        project = 'runs/train',
        name = 'exp',
        freeze = 0
        )


