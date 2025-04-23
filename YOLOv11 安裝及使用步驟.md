# YOLOv11 安裝及使用步驟

## 環境需求
- Windows 10/11
- Python 3.11
- NVIDIA GPU (測試環境：RTX 4060 Ti)
- CUDA 12.1

## 安裝步驟

1. 創建並啟動虛擬環境
```bash
python -m venv .venv
.\.venv\Scripts\activate
```

2. 安裝必要套件
```bash
# 創建 requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics opencv-python numpy
```

## 測試代碼

創建 `test_yolo.py` 文件：

```python
from ultralytics import YOLO
import torch

def main():
    # 檢查 CUDA 是否可用
    print("=== GPU 信息 ===")
    print(f"CUDA 是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"當前使用的 GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"GPU 數量: {torch.cuda.device_count()}")
        print(f"當前 GPU 內存使用情況: {torch.cuda.memory_allocated(0) / 1024 ** 2:.2f} MB")
        print(f"當前 GPU 內存快取: {torch.cuda.memory_reserved(0) / 1024 ** 2:.2f} MB")
    print("==============")
    
    # 載入 YOLOv11 模型
    print("\n正在下載並載入模型...")
    model = YOLO('yolo11n.pt')
    
    # 使用 GPU 進行推理
    print("\n開始進行圖像推理...")
    results = model('https://ultralytics.com/images/bus.jpg', device=0)
    
    # 顯示結果
    print("\n=== 檢測結果 ===")
    for result in results:
        print(f"檢測到的物件數量: {len(result.boxes)}")
        for box in result.boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            print(f"類別: {result.names[class_id]}, 信心度: {confidence:.2f}")

if __name__ == "__main__":
    main()
```

## 運行測試

```bash
python test_yolo.py
```

## 測試結果

使用 RTX 4060 Ti 的測試結果：

```
=== GPU 信息 ===
CUDA 是否可用: True
當前使用的 GPU: NVIDIA GeForce RTX 4060 Ti
CUDA 版本: 12.1
GPU 數量: 1
當前 GPU 內存使用情況: 0.00 MB
當前 GPU 內存快取: 0.00 MB
==============

正在下載並載入模型...

開始進行圖像推理...
image 1/1 C:\Users\dan\Desktop\Project\114_yolov11\bus.jpg: 640x480 4 persons, 1 bus, 43.2ms
Speed: 1.5ms preprocess, 43.2ms inference, 51.0ms postprocess per image at shape (1, 3, 640, 480)

=== 檢測結果 ===
檢測到的物件數量: 5
類別: bus, 信心度: 0.94
類別: person, 信心度: 0.89
類別: person, 信心度: 0.88
類別: person, 信心度: 0.86
類別: person, 信心度: 0.62
```

## 性能分析

- 預處理時間：1.5ms
- 推理時間：43.2ms
- 後處理時間：51.0ms
- 總處理時間：約 95.7ms

## 注意事項

1. 確保已安裝 NVIDIA 顯示卡驅動程式
2. 確保 CUDA 版本與 PyTorch 版本相匹配
3. 如果遇到 CUDA 不可用的問題，可能需要重新安裝 PyTorch CUDA 版本 