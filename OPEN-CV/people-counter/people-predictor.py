import cv2
import time
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO

VIDEO_PATH = "background video ｜ people ｜ walking ｜.mp4"
WEIGHTS = "yolov8n.pt"
USE_BYTETRACK = True
CONF_THRESH = 0.3
IMG_W, IMG_H = 1280, 720
LINE_Y = 360
DRAW_TRAILS = True
MAX_TRAIL = 20

model = YOLO(WEIGHTS)
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

last_center_y = {}
trail_history = defaultdict(lambda: deque(maxlen=MAX_TRAIL))
count_up = 0
count_down = 0
seen_once_block = set()

t0 = time.time()
frame_count = 0

_ = model.predict(np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8), verbose=False)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame_count += 1
    frame = cv2.resize(frame, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)

    if USE_BYTETRACK:
        results = model.track(frame, tracker="bytetrack.yaml", persist=True, verbose=False, conf=CONF_THRESH)
    else:
        results = model(frame, verbose=False, conf=CONF_THRESH)

    boxes = getattr(results[0], "boxes", None)
    if boxes is None:
        cv2.line(frame, (0, LINE_Y), (IMG_W, LINE_Y), (0, 255, 255), 3)
        cv2.imshow("People Counter (YOLOv8+ByteTrack)", frame)
        if cv2.waitKey(1) == 27:
            break
        continue

    xyxy = boxes.xyxy.cpu().numpy() if boxes.xyxy is not None else np.empty((0,4))
    cls  = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.empty((0,), dtype=int)
    conf = boxes.conf.cpu().numpy() if boxes.conf is not None else np.empty((0,))
    ids  = boxes.id
    ids = ids.cpu().numpy().astype(int) if ids is not None else None

    cv2.line(frame, (0, LINE_Y), (IMG_W, LINE_Y), (0, 255, 255), 3)
    cv2.putText(frame, "COUNT LINE", (20, LINE_Y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

    for i in range(len(xyxy)):
        if cls[i] != 0:
            continue

        x1, y1, x2, y2 = xyxy[i].astype(int)
        w, h = x2 - x1, y2 - y1
        if w < 15 or h < 30:
            continue

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        obj_id = None
        if ids is not None and i < len(ids):
            obj_id = int(ids[i])

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 230, 0), 2)
        label = f"ID {obj_id}" if obj_id is not None else "person"
        if conf is not None and i < len(conf):
            label += f" {conf[i]:.2f}"
        cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2, cv2.LINE_AA)

        if DRAW_TRAILS and obj_id is not None:
            trail_history[obj_id].append((cx, cy))
            pts = list(trail_history[obj_id])
            for j in range(1, len(pts)):
                cv2.line(frame, pts[j-1], pts[j], (255, 255, 255), 2)

        if obj_id is not None:
            prev_y = last_center_y.get(obj_id, None)
            last_center_y[obj_id] = cy

            if prev_y is not None:
                if prev_y < LINE_Y <= cy:
                    if obj_id not in seen_once_block:
                        count_down += 1
                        seen_once_block.add(obj_id)
                elif prev_y > LINE_Y >= cy:
                    if obj_id not in seen_once_block:
                        count_up += 1
                        seen_once_block.add(obj_id)

            if abs(cy - LINE_Y) > 60 and obj_id in seen_once_block:
                seen_once_block.remove(obj_id)

        cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

    elapsed = time.time() - t0
    fps = frame_count / elapsed if elapsed > 0 else 0.0

    hud = f"UP: {count_up}   DOWN: {count_down}   TOTAL: {count_up + count_down}   FPS: {fps:.1f}"
    (tw, th), _ = cv2.getTextSize(hud, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(frame, (10, 10), (10 + tw + 14, 10 + th + 14), (0, 0, 0), -1)
    cv2.putText(frame, hud, (17, 10 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("People Counter (YOLOv8 + ByteTrack)", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
print(f"Final UP: {count_up}  DOWN: {count_down}  TOTAL: {count_up + count_down}")
