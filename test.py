import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
model_path = r"D:/碩士班/程式測驗/ultralytics-main/runs/train/exp10/weights/best.pt" 
data_path = r"D:/碩士班/程式測驗/ultralytics-main/ultralytics/cfg/datasets/african-wildlife.yaml"

if __name__ == '__main__':
    # 載入訓練好的模型
    model = YOLO(model_path)

    # 進行測試
    metrics = model.val(
        data = data_path
        batch = 16,               # 測試 batch size
        imgsz = 640,              # 測試圖片大小
        save_json = True,         # 產生 JSON 格式結果（COCO 格式）
        conf = 0.492               # 置信度閾值
    )

    # 顯示測試結果
    print(metrics)