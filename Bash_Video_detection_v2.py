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
            with open('target_classes.csv', 'r', encoding='utf-8') as csvfile:
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
    
    def load_roi_polygon(self, path="roi_polygon.txt"):
        with open(path, "r") as f:
            lines = f.readlines()
            points = [tuple(map(int, line.strip().split(","))) for line in lines]
        return np.array(points, dtype=np.int32)


    def apply_roi_mask(self, frame):
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        polygon = self.load_roi_polygon()

        cv2.fillPoly(mask, [polygon.astype(np.int32)], 255)

        # 複製一張畫面並套用遮罩
        roi_only = cv2.bitwise_and(frame, frame, mask=mask)

        return roi_only, mask, polygon


    def get_stream_url(self):
        """獲取 YouTube 直播串流的 URL"""
        with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
            info = ydl.extract_info(self.youtube_url, download=False)
            return info['url']
    
    def preprocess_frame(self, frame):
        """預處理影像幀以符合模型輸入要求"""
        if frame is None:
            return None
            
        # 調整影像大小為模型要求的尺寸
        # frame_resized = cv2.resize(frame, (640, 640))
        frame_resized = frame 
        
        # 轉換顏色空間從 BGR 到 RGB
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        # 正規化像素值到 [0,1]
        frame_normalized = frame_rgb.astype(np.float32) / 255.0
        
        # 調整維度順序從 HWC 到 BCHW
        frame_chw = np.transpose(frame_normalized, (2, 0, 1))
        frame_bchw = np.expand_dims(frame_chw, axis=0)
        
        return frame_bchw
    
    def process_frame(self, frame):
        if frame is None:
            return None, False
        
        # Step 1: 切出下半部 (1920x640)，假設影像是 1920x1080
        h, w = frame.shape[:2]
        crop_y_start = h - 640
        cropped = frame[crop_y_start:h, 0:w]  # (640, 1920, 3)

        # Step 2: 切割為三塊 (640x640)
        slices = [cropped[:, i*640:(i+1)*640] for i in range(3)]
        detection_results = []     

        for slice_img in slices:
            with torch.cuda.amp.autocast() if self.device == 'cuda' else torch.no_grad():
                result = self.model(slice_img, conf=self.detection_threshold)
                detection_results.append(result)

        # 繪製結果（分開做）
        annotated_imgs = [cv2.cvtColor(r[0].plot(), cv2.COLOR_RGB2BGR) for r in detection_results]
        combined_bottom = cv2.hconcat(annotated_imgs)

        # # Step 4: 拼接三張回成 1920x640 圖片
        # combined_bottom = cv2.hconcat(results)

        # Step 5: 把偵測後的 1920x640 蓋回原始 frame 下方
        output_frame = frame.copy()
        output_frame[crop_y_start:h, 0:w] = combined_bottom

        # 檢查是否有偵測到目標類別
        has_target = any(
            any(self.is_target_class(int(box.cls[0])) for box in result[0].boxes)
            for result in detection_results
        )
            
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
                self.frame_buffer.append(frame)
                if len(self.frame_buffer) > self.buffer_size:
                    self.frame_buffer.pop(0)
            return output_frame, False
    
    def save_detection(self):
        if not self.frame_buffer:
            return
            
        # 產生時間戳記
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join("detections", f"{self.modelselect[:-3]}_{self.video_source}_{timestamp}.mp4")
        
        # 設定影片寫入器
        height, width = self.frame_buffer[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))
        
        # 寫入緩衝區中的幀
        for frame in self.frame_buffer:
            out.write(frame)
        
        out.release()
        print(f"已儲存偵測結果到: {output_path}")
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
                # if self.is_detecting:
                #     self.frame_buffer.append(processed_frame)
                
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
            if 'cap' in locals():
                cap.release()
            # cv2.destroyAllWindows()

if __name__ == "__main__":

    Video_path = './TestVideo/'

    model_list = ['yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt']

    for modelselect in model_list:

        for video in os.listdir(Video_path):
            local_video_path = os.path.join(Video_path, video)
            
            print(local_video_path)
            detector = YouTubeObjectDetector(local_video_path=local_video_path,modelselect=modelselect, video_source=video)
            
            detector.run() 