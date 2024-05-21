from ultralytics import YOLO
import cv2
from deep_sort.deep_sort import DeepSort
import numpy as np

# YOLO 및 Deep SORT 모델 로딩
yolo_model = YOLO("yolov8n.pt")
deep_sort_weights = "deep_sort/deep/checkpoint/ckpt.t7"
tracker = DeepSort(model_path=deep_sort_weights, max_age=50)

# 비디오 및 출력 설정
video_path = "images/up2.mp4"
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
output_path = "output.mp4"
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# 중앙선 및 초기 변수 설정
line_start = (0, frame_height // 2)
line_end = (frame_width, frame_height // 2)
in_count, exit_count = 0, 0
prev_center_y = {}

while True:
    ret, frame = cap.read()

    if not ret:
        break

    og_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = og_frame.copy()

    # YOLO를 사용하여 객체 검출
    results = yolo_model(frame, device=0, classes=0, conf=0.8)

    # 추적을 위한 배열 초기화
    bboxes, confidences = [], []

    class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    for result in results:
        boxes = result.boxes
        conf = boxes.conf
        xywh = boxes.xywh
        cls = boxes.cls.tolist()

        # Deep SORT를 위한 정보 저장
        confidences.append(conf.detach().cpu().numpy())
        bboxes.append(xywh.cpu().numpy())

        for class_index in cls:
            class_name = class_names[int(class_index)]

    if bboxes:
        confidences = np.concatenate(confidences)
        bboxes = np.concatenate(bboxes)

        # Deep SORT를 사용하여 객체 추적
        tracks = tracker.update(bboxes, confidences, og_frame)
        
        inside_count = 0  # 현재 안에 있는 객체 수를 초기화
        for track in tracks:
            track_id = int(track[4])  # track의 5번째 값은 track_id
            x1, y1, x2, y2 = track[0:4]  # track의 처음 4개 값은 좌표
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # 'in' 및 'exit' 카운트
            if track_id in prev_center_y:
                if prev_center_y[track_id] < line_start[1] and center_y >= line_start[1]:
                    exit_count += 1
                elif prev_center_y[track_id] >= line_start[1] and center_y < line_start[1]:
                    in_count += 1

            prev_center_y[track_id] = center_y

            # 트랙 ID에 따른 색상 설정
            color_id = track_id % 3
            color = [(0, 0, 255), (255, 0, 0), (0, 255, 0)][color_id]
            
             # 객체 주변에 바운딩 박스 그리기
            cv2.rectangle(og_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            # 객체 중심에 점 표시
            cv2.circle(og_frame, (center_x, center_y), 3, color, -1)
        
        
        # 중앙선 그리기
        cv2.line(og_frame, line_start, line_end, (0, 0, 255), 2)
        
        # 'in' 및 'exit' 카운트 표시
        cv2.putText(og_frame, f"In : {in_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(og_frame, f"Exit : {exit_count}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # in_count에서 exit_count를 뺀 값 출력
        current_inside_count = in_count - exit_count
        cv2.putText(og_frame, f"Total Inside Count: {current_inside_count}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # 프레임 및 비디오 출력
        out.write(cv2.cvtColor(og_frame, cv2.COLOR_RGB2BGR))
        cv2.imshow("Video", og_frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 자원 해제
cap.release()
out.release()
cv2.destroyAllWindows()
