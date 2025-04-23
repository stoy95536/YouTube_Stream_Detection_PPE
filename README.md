# YouTube 直播物件偵測系統

這個專案使用 YOLOv11 模型來偵測 YouTube 直播串流中的物件，並在偵測到物件時自動儲存影片片段。

## 功能特點

- 即時處理 YouTube 直播串流
- 使用 YOLOv11 進行物件偵測
- 自動儲存包含偵測到物件的影片片段
- 支援往前回溯5秒的影片內容
- 可自訂偵測閾值和輸出目錄

## 安裝需求

1. 安裝 Python 3.8 或更新版本
2. 安裝所需套件：
```bash
pip install -r requirements.txt
```

## 使用方法

1. 修改 `youtube_detection.py` 中的 YouTube URL：
```python
youtube_url = "你的 YouTube 直播 URL"
```

2. 執行程式：
```bash
python youtube_detection.py
```

3. 操作說明：
- 程式會自動開始處理直播串流
- 當偵測到物件時，會自動儲存包含該物件的影片片段
- 按 'q' 鍵可以退出程式

## 輸出

- 偵測到的影片片段會儲存在 `detections` 目錄中
- 檔案名稱格式為：`detection_YYYYMMDD_HHMMSS.mp4`

## 注意事項

- 確保網路連線穩定
- 建議使用 GPU 來加速處理
- 程式會自動下載 YOLOv11 模型（首次執行時） 