import cv2
import numpy as np

points = []

def mouse_callback(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"選取點: ({x}, {y})")

def main():
    global points
    # 讀取影像或任一幀
    # frame = cv2.imread("sample_frame.jpg")  # ← 你可換成影片一幀擷取圖

    cap = cv2.VideoCapture("./Video/live_20250422_143004.mp4")
    ret, frame = cap.read()
    cap.release()
    # frame = cv2.VideoCapture("./Video/live_20250422_143004.mp4").read()  # ← 你可換成影片一幀擷取圖

    clone = frame.copy()
    cv2.namedWindow("選取 ROI 多邊形區域")
    cv2.setMouseCallback("選取 ROI 多邊形區域", mouse_callback)

    while True:
        display = clone.copy()
        for p in points:
            cv2.circle(display, p, 5, (0, 0, 255), -1)
        if len(points) > 1:
            cv2.polylines(display, [np.array(points)], isClosed=True, color=(0, 255, 255), thickness=2)

        cv2.imshow("選取 ROI 多邊形區域", display)
        key = cv2.waitKey(1)

        if key == 13:  # 按 Enter 完成
            break
        elif key == 27:  # 按 Esc 退出
            points = []
            break

    cv2.destroyAllWindows()

    if points:
        print("\n🟢 選取完成，ROI 座標如下：")
        print(points)

        # 可輸出為檔案
        with open("roi_polygon.txt", "w") as f:
            for pt in points:
                f.write(f"{pt[0]},{pt[1]}\n")
        print("已儲存為 roi_polygon.txt")

if __name__ == "__main__":
    main()