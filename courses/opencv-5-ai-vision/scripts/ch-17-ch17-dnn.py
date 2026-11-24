#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCV 5 والرؤية الحاسوبية بالذكاء الاصطناعي — التعلم العميق: cv2.dnn وONNX وYOLO
OpenCV 5 and AI Computer Vision — Dnn

Book slug : opencv-5-ai-vision
Chapter   : ch-17-ch17-dnn (lesson 17)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/opencv-5-ai-vision/code/ch-17-ch17-dnn.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCV 5 والرؤية الحاسوبية بالذكاء الاصطناعي — التعلم العميق: cv2.dnn وONNX وYOLO
OpenCV 5 and AI Computer Vision — Dnn

Book slug : opencv-5-ai-vision
Chapter   : ch-17-ch17-dnn (lesson 17)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/chapters/ch17_dnn.md
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

net = cv2.dnn.readNetFromONNX("models/yolov8n.onnx")
input_size = (640, 640)

# ────────────────────────────────────────────────────────────────────────
# [002] content/chapters/ch17_dnn.md
# ────────────────────────────────────────────────────────────────────────
blob = cv2.dnn.blobFromImage(img, 1/255.0, input_size, swapRB=True, crop=False)
net.setInput(blob)
outputs = net.forward()
print("شكل المخرجات:", outputs.shape)

# ────────────────────────────────────────────────────────────────────────
# [003] content/chapters/ch17_dnn.md
# ────────────────────────────────────────────────────────────────────────
def postprocess(output, img, conf_thr=0.45):
    h, w = img.shape[:2]
    x_factor, y_factor = w / input_size[0], h / input_size[1]

    boxes, scores, class_ids = [], [], []
    rows = output.reshape(-1, output.shape[-1])
    for row in rows:
        conf = row[4]
        if conf < conf_thr:
            continue

        class_id = int(np.argmax(row[5:]))
        cx, cy, bw, bh = row[:4]
        x1 = int((cx - bw/2) * x_factor)
        y1 = int((cy - bh/2) * y_factor)
        boxes.append([x1, y1, int(bw*x_factor), int(bh*y_factor)])
        scores.append(float(conf))
        class_ids.append(class_id)

    # NMS: إزالة الصناديق المتراكبة
    idxs = cv2.dnn.NMSBoxes(boxes, scores, conf_thr, 0.25)
    return [(boxes[i], class_ids[i], scores[i]) for i in np.array(idxs).flatten()]

# ────────────────────────────────────────────────────────────────────────
# [004] content/chapters/ch17_dnn.md
# ────────────────────────────────────────────────────────────────────────
LABELS = open("models/coco.names").read().strip().split("\n")

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ok, frame = cap.read()
    if not ok: break

    blob = cv2.dnn.blobFromImage(frame, 1/255.0, input_size,
                                 swapRB=True, crop=False)
    net.setInput(blob)
    out = net.forward()

    for (box, cid, sc) in postprocess(out, frame):
        x, y, bw, bh = box
        cv2.rectangle(frame, (x, y), (x+bw, y+bh), (0, 255, 0), 2)
        cv2.putText(frame, f"{LABELS[cid]} {sc:.2f}", (x, y-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    cv2.imshow("YOLO with OpenCV", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release(); cv2.destroyAllWindows()

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/opencv-5-ai-vision/notebooks/ch-17-ch17-dnn.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np

net = cv2.dnn.readNetFromONNX("models/yolov8n.onnx")
input_size = (640, 640)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/opencv-5-ai-vision/notebooks/ch-17-ch17-dnn.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
blob = cv2.dnn.blobFromImage(img, 1/255.0, input_size, swapRB=True, crop=False)
net.setInput(blob)
outputs = net.forward()
print("شكل المخرجات:", outputs.shape)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/opencv-5-ai-vision/notebooks/ch-17-ch17-dnn.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
def postprocess(output, img, conf_thr=0.45):
    h, w = img.shape[:2]
    x_factor, y_factor = w / input_size[0], h / input_size[1]

    boxes, scores, class_ids = [], [], []
    rows = output.reshape(-1, output.shape[-1])
    for row in rows:
        conf = row[4]
        if conf < conf_thr:
            continue

        class_id = int(np.argmax(row[5:]))
        cx, cy, bw, bh = row[:4]
        x1 = int((cx - bw/2) * x_factor)
        y1 = int((cy - bh/2) * y_factor)
        boxes.append([x1, y1, int(bw*x_factor), int(bh*y_factor)])
        scores.append(float(conf))
        class_ids.append(class_id)

    # NMS: إزالة الصناديق المتراكبة
    idxs = cv2.dnn.NMSBoxes(boxes, scores, conf_thr, 0.25)
    return [(boxes[i], class_ids[i], scores[i]) for i in np.array(idxs).flatten()]

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/opencv-5-ai-vision/notebooks/ch-17-ch17-dnn.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
LABELS = open("models/coco.names").read().strip().split("\n")

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ok, frame = cap.read()
    if not ok: break

    blob = cv2.dnn.blobFromImage(frame, 1/255.0, input_size,
                                 swapRB=True, crop=False)
    net.setInput(blob)
    out = net.forward()

    for (box, cid, sc) in postprocess(out, frame):
        x, y, bw, bh = box
        cv2.rectangle(frame, (x, y), (x+bw, y+bh), (0, 255, 0), 2)
        cv2.putText(frame, f"{LABELS[cid]} {sc:.2f}", (x, y-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    cv2.imshow("YOLO with OpenCV", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release(); cv2.destroyAllWindows()
