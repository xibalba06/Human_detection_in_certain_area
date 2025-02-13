import torch
import cv2
import numpy as np

# YOLOv5 modelini yükle
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x' seçenekleri mevcut

# Video veya kamera kaynağını aç
cap = cv2.VideoCapture(0)  # 0, varsayılan kamerayı açar. Video dosyası için 'video.mp4' gibi bir yol verin.

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Kamera görüntüsü alınamadı.")
        break

    # YOLOv5 ile nesne tespiti
    results = model(frame)

    # Tespit edilen nesneleri çizin
    for *box, conf, cls in results.xyxy[0]:  # Tespit edilen nesneler
        x1, y1, x2, y2 = map(int, box)  # Koordinatları al
        label = f"{model.names[int(cls)]} {conf:.2f}"  # Etiket ve güven skoru
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Dikdörtgen çiz
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Etiketi yaz

    # Görüntüyü göster
    cv2.imshow('YOLOv5 Object Detection', frame)

    # 'q' tuşuna basarak çıkış yap
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()