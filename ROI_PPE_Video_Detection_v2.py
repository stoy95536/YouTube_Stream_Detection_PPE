import cv2
import numpy as np
from ultralytics import YOLO
import yt_dlp
import time
from datetime import datetime
import os
import torch
import csv

class YouTubeObjectDetector:
    def __init__(self, youtube_url=None, local_video_path=None, modelselect=None, video_source=None, detection_threshold=0.5, buffer_size=150):
        self.youtube_url = youtube_url
        self.local_video_path = local_video_path
        self.detection_threshold = detection_threshold
        self.buffer_size = buffer_size
        self.frame_buffer = []
        self.is_detecting = False
        self.last_detection_time = time.time()
        self.detection_start_time = None
        self.modelselect = modelselect
        self.video_source = video_source
        
        # 檢查是否有可用的 GPU
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"使用設備: {self.device}")
        
        if self.device == 'cuda':
            # GPU 優化設定
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.cuda.empty_cache()
            
            # 設定 CUDA 記憶體分配器
            torch.cuda.set_per_process_memory_fraction(0.8)  # 使用 80% 的 GPU 記憶體
            torch.cuda.memory.set_per_process_memory_fraction(0.8)
        
        # 載入模型
        self.model = YOLO(self.modelselect)
        if self.device == 'cuda':
            self.model.to(self.device)
            # 啟用 FP16 自動混合精度
            self.model.model.half()
        
        # 載入目標類別
        self.target_classes = self.load_target_classes()
        
        # 設定 yt-dlp 選項
        self.ydl_opts = {
            'format': 'best[ext=mp4]',
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
        }
        
        # 根據設備設定批次處理大小
        self.batch_size = 8 if self.device == 'cuda' else 1
        
        # 建立輸出目錄
        if not os.path.exists("detections"):
            os.makedirs("detections")
    
    def load_target_classes(self):
        """載入目標類別設定"""
        target_classes = set()
        try:
            with open('ppe_target_classes.csv', 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    target_classes.add(int(row['類別ID']))
            print(f"已載入 {len(target_classes)} 個目標類別")
            return target_classes
        except Exception as e:
            print(f"載入目標類別時發生錯誤: {e}")
            return set()
    
    def is_target_class(self, class_id):
        """檢查是否為目標類別"""
        return class_id in self.target_classes
    
    def load_roi_polygon(self, path="roi_ppe_polygon.txt"):
        with open(path, "r") as f:
            lines = f.readlines()
            points = [tuple(map(int, line.strip().split(","))) for line in lines]
        return np.array(points, dtype=np.int32)

    def get_stream_url(self):
        """獲取 YouTube 直播串流的 URL"""
        with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
            info = ydl.extract_info(self.youtube_url, download=False)
            return info['url']

    def apply_roi_mask(self, frame):
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        polygon = self.load_roi_polygon()
        cv2.fillPoly(mask, [polygon.astype(np.int32)], 255)
        # 複製一張畫面並套用遮罩
        roi_only = cv2.bitwise_and(frame, frame, mask=mask)
        return roi_only, mask, polygon
    
    def is_inside_roi(self, bbox, polygon, slice_index=0):
        """
        檢查檢測框是否在ROI內
        bbox: [x1, y1, x2, y2] - 相對於切片的座標
        polygon: ROI多邊形
        slice_index: 切片索引 (0, 1, 2)
        """
        # 將bbox從切片座標轉換為裁切後完整影像座標
        x1, y1, x2, y2 = bbox
        x_offset = slice_index * 640
        x1 += x_offset
        x2 += x_offset
        
        # 計算bbox中心點
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # 檢查中心點是否在多邊形內
        return cv2.pointPolygonTest(polygon, (center_x, center_y), False) >= 0

    def process_frame(self, frame):
        if frame is None:
            return None, False
        
        original_h, original_w = frame.shape[:2]
        crop_y_start = original_h - 640
        
        # 載入ROI多邊形
        roi_polygon = self.load_roi_polygon()
        
        # 處理裁切的畫面
        cropped = frame[crop_y_start:original_h, 0:original_w].copy()
        cropped_h, cropped_w = cropped.shape[:2]
        
        # 在完整畫面上繪製ROI
        frame_with_roi = frame.copy()
        cv2.polylines(frame_with_roi, [roi_polygon], isClosed=True, color=(0, 255, 255), thickness=2)
        
        detection_results_list = []
        slices = [cropped[:, i*640:(i+1)*640] for i in range(3)]
        
        # 對每個切片進行檢測
        for slice_img in slices:
            with torch.cuda.amp.autocast() if self.device == 'cuda' else torch.no_grad():
                # 模型會自動將輸入的640x640調整為800x800
                result = self.model(slice_img, conf=self.detection_threshold)
                detection_results_list.append(result)
        
        annotated_slices = []
        has_target = False
        has_target_in_roi = False
        
        # 處理每個切片的檢測結果
        for i, result in enumerate(detection_results_list):
            annotated_slice = slices[i].copy()
            
            # 繪製該切片的ROI部分
            slice_roi_polygon = roi_polygon.copy()
            # 調整ROI多邊形座標以適應切片
            slice_roi_polygon[:, 0] = slice_roi_polygon[:, 0] - i*640
            # 只保留在當前切片範圍內的點
            mask = (slice_roi_polygon[:, 0] >= 0) & (slice_roi_polygon[:, 0] < 640)
            if any(mask):  # 如果有點在當前切片內
                valid_roi = slice_roi_polygon[mask]
                if len(valid_roi) >= 3:  # 需要至少3個點才能形成多邊形
                    cv2.polylines(annotated_slice, [valid_roi], isClosed=False, color=(0, 255, 255), thickness=2)
            
            for *xyxy, conf, cls in result[0].boxes.data.tolist():
                x1, y1, x2, y2 = map(int, xyxy)
                PAD = (800 - 640) // 2  # = 80
                
                # 如果模型輸入被調整為800x800，需要將座標調整回640x640
                # 計算縮放比例
                scale_factor = 640 / 800
                x1 = int((x1 - PAD) * scale_factor)
                y1 = int((y1 - PAD) * scale_factor)
                x2 = int((x2 - PAD) * scale_factor)
                y2 = int((y2 - PAD) * scale_factor)
                
                label = f'{self.model.names[int(cls)]} {conf:.2f}'
                
                # 檢查是否為目標類別
                is_target = self.is_target_class(int(cls))
                if is_target:
                    has_target = True
                    
                    # 檢查目標是否在ROI內
                    in_roi = self.is_inside_roi([x1, y1, x2, y2], roi_polygon, i)
                    if in_roi:
                        has_target_in_roi = True
                        # 如果在ROI內，使用紅色繪製
                        color = (0, 0, 255)  # 紅色 (BGR)
                        cv2.rectangle(annotated_slice, (x1, y1), (x2, y2), color, 2)
                        # 添加警告標籤
                        warning_label = "WARNING: In Danger Zone!"
                        cv2.putText(annotated_slice, warning_label, (x1, y2 + 20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    else:
                        # 如果不在ROI內，使用綠色繪製
                        color = (0, 255, 0)  # 綠色 (BGR)
                        cv2.rectangle(annotated_slice, (x1, y1), (x2, y2), color, 2)
                else:
                    # 非目標類別，使用綠色繪製
                    color = (0, 255, 0)  # 綠色 (BGR)
                    cv2.rectangle(annotated_slice, (x1, y1), (x2, y2), color, 2)
                
                cv2.putText(annotated_slice, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            annotated_slices.append(annotated_slice)
        
        # 組合所有切片
        combined_bottom = np.hstack(annotated_slices)
        
        # 將處理後的底部部分放回完整畫面
        output_frame = frame_with_roi.copy()
        output_frame[crop_y_start:original_h, 0:original_w] = combined_bottom
        
        # 如果有在ROI內的目標物，添加全局警告
        if has_target_in_roi:
            warning_text = "ALERT: Person in Danger Zone Detected!"
            cv2.putText(output_frame, warning_text, (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
        # 檢測緩衝區的處理
        if has_target:
            if not self.is_detecting:
                self.is_detecting = True
                self.detection_start_time = time.time()
                self.frame_buffer = self.frame_buffer[-self.buffer_size:]
            self.last_detection_time = time.time()
            return output_frame, True
        else:
            if self.is_detecting and time.time() - self.last_detection_time > 5:
                self.is_detecting = False
                self.save_detection()
                self.frame_buffer = []
            elif not self.is_detecting:
                self.frame_buffer.append(output_frame)
                if len(self.frame_buffer) > self.buffer_size:
                    self.frame_buffer.pop(0)
            return output_frame, False
    
    def save_detection(self):
        if not self.frame_buffer:
            print("⚠️ frame_buffer 為空，無資料可儲存")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join("detections", f"{self.modelselect[:-3]}_{self.video_source}_{timestamp}.mp4")
        
        height, width = self.frame_buffer[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

        if not out.isOpened():
            print("⚠️ 無法開啟 VideoWriter，檢查路徑或格式")
            return

        for i, frame in enumerate(self.frame_buffer):
            if frame is not None and frame.size > 0:
                if frame.shape[:2] != (height, width):
                    print(f"⚠️ 第 {i} 幀大小不一致，跳過")
                    continue
                out.write(frame)
            else:
                print(f"⚠️ 第 {i} 幀為空或無效，跳過")

        out.release()
        print(f"✅ 偵測影片儲存於: {output_path}")
        self.frame_buffer = []
    
    def run(self):
        try:
            
            if self.local_video_path:
                print(f"使用本地影片：{self.local_video_path}")
                cap = cv2.VideoCapture(self.local_video_path)
            else:
                # 獲取直播串流 URL
                stream_url = self.get_stream_url()
                print("成功獲取直播串流 URL")
            
                # 使用 OpenCV 讀取串流
                cap = cv2.VideoCapture(stream_url)

            if not cap.isOpened():
                print("無法開啟影片串流")
                return
                
            print("開始處理影片串流...")
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("無法讀取影片幀")
                    break
                
                # 處理當前幀
                processed_frame, has_detections = self.process_frame(frame)
                if processed_frame is None:
                    continue
                
                # 如果正在偵測中，將幀加入緩衝區
                if self.is_detecting:
                    self.frame_buffer.append(processed_frame)


                # 顯示結果
                
                # cv2.imshow('YouTube Detection', processed_frame)
                
                # 按 'q' 退出
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     # 如果正在偵測中，儲存最後的偵測結果
                #     if self.is_detecting:
                #         self.save_detection()
                #     break
                
        except Exception as e:
            print(f"發生錯誤: {str(e)}")
            if self.is_detecting:
                self.save_detection()
            
        finally:
            if self.is_detecting or self.frame_buffer:
                print("影片結束，自動儲存最後的偵測結果")
                self.save_detection()
            if 'cap' in locals():
                cap.release()
            # cv2.destroyAllWindows()

if __name__ == "__main__":
    Video_path = './TestVideo/'
    model_list = ['ppe.pt']

    for modelselect in model_list:
        print(f"\n----- 開始處理模型：{modelselect} -----")
        for i in range(1, 43):
            video_filename = f'Video_{i}.mp4'
            local_video_path = os.path.join(Video_path, video_filename)

            print(f"嘗試開啟影片：{local_video_path}")
            detector = YouTubeObjectDetector(local_video_path=local_video_path, modelselect=modelselect, video_source=video_filename)
            cap = cv2.VideoCapture(local_video_path)

            if not cap.isOpened():
                print(f"⚠️ 無法開啟影片：{local_video_path}")
                cap.release()
                continue  # 跳過無法開啟的影片

            print(f"成功開啟影片：{local_video_path}")
            cap.release() # 這裡釋放 cap，讓 detector 內部重新開啟
            detector.run()
            time.sleep(5)
    
