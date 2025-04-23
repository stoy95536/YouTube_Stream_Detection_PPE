from ultralytics import YOLO
import torch
import csv
from datetime import datetime

def list_yolo_classes():
    # 檢查是否有可用的 GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用 {device} 進行運算")
    
    # 載入模型
    model = YOLO('yolo11n.pt')
    
    # 獲取類別名稱
    class_names = model.names
    
    # 顯示類別數量
    print(f"\nYOLOv11 可以偵測 {len(class_names)} 種物件類別：")
    print("-" * 50)
    
    # 按照類別 ID 排序並顯示
    for class_id, class_name in sorted(class_names.items()):
        print(f"類別 ID: {class_id:3d} | 類別名稱: {class_name}")
    
    print("-" * 50)
    print(f"總共 {len(class_names)} 種物件類別")
    
    # 儲存為 CSV 檔案
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"yolo_classes_{timestamp}.csv"
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # 寫入標題
        writer.writerow(['類別ID', '類別名稱'])
        # 寫入資料
        for class_id, class_name in sorted(class_names.items()):
            writer.writerow([class_id, class_name])
    
    print(f"\n類別資訊已儲存至 {filename}")

if __name__ == "__main__":
    list_yolo_classes() 