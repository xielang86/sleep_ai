import cv2,time
import mediapipe as mp

# 初始化FaceMesh模型，启用refine_landmarks以获取虹膜关键点
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# 初始化绘图工具
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# 打开摄像头
cap = cv2.VideoCapture(0)
n = 0
while cap.isOpened() and n < 30:
    success, image = cap.read()
    if not success:
        print("无法读取摄像头画面。")
        continue

    # 将图像颜色空间从BGR转换为RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)

    # 将图像颜色空间转换回BGR以便使用OpenCV显示
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # 绘制左眼眼眶
            left_eye_landmarks = [face_landmarks.landmark[i] for i in range(33, 134)]
            for landmark in left_eye_landmarks:
                ih, iw, _ = image.shape
                x, y = int(landmark.x * iw), int(landmark.y * ih)
                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

            # 绘制左眼虹膜
            left_iris_landmarks = [face_landmarks.landmark[i] for i in range(468, 473)]
            for landmark in left_iris_landmarks:
                ih, iw, _ = image.shape
                x, y = int(landmark.x * iw), int(landmark.y * ih)
                cv2.circle(image, (x, y), 2, (0, 0, 255), -1)

            # 绘制右眼眼眶
            right_eye_landmarks = [face_landmarks.landmark[i] for i in range(263, 363)]
            for landmark in right_eye_landmarks:
                ih, iw, _ = image.shape
                x, y = int(landmark.x * iw), int(landmark.y * ih)
                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

            # 绘制右眼虹膜
            right_iris_landmarks = [face_landmarks.landmark[i] for i in range(473, 478)]
            for landmark in right_iris_landmarks:
                ih, iw, _ = image.shape
                x, y = int(landmark.x * iw), int(landmark.y * ih)
                cv2.circle(image, (x, y), 2, (0, 0, 255), -1)

    # 显示图像
    cv2.imshow('Eye Landmarks', image)

    # 按 'q' 键退出循环
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
    n += 1
    time.sleep(1)

# 释放摄像头资源并关闭所有窗口
cap.release()
cv2.destroyAllWindows()