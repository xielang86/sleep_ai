import time
from algorithm import pose
import mediapipe as mp
import cv2,os

# 初始化mediapipe的手部解决方案
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
mp_drawing = mp.solutions.drawing_utils

# 打开摄像头
cap = cv2.VideoCapture(0)

detector = pose.PoseDetector()
n = 0
while cap.isOpened() and n < 30:
  ret, frame = cap.read()
  if not ret:
    print("read failed!")
    break
  # 将图像从BGR格式转换为RGB格式（mediapipe要求的输入格式）
  image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  results = face_mesh.process(image)
  # try: 
  #   result = detector.Detect(image)
  #   if result == None:
  #     print("failed to detect")
  #   else:
  #     print(result.leftEyeClosed, result.rightEyeClosed)
  #     print(result.pose_type.name)
  # except Exception as e:
  #   # 捕捉任何异常并抛出
  #   print(e)
  #   raise e
  if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
      # 这里可以根据眼睛的关键点计算EAR等指标来判断眼睛状态
      # 简化示例，仅绘制关键点
      try:
        mp_drawing.draw_landmarks(
        image=image,
        landmark_list=face_landmarks,
        # connections=mp_face_mesh.FACEMESH_TESSELATION,
        connections=mp_face_mesh.FACEMESH_IRISES,
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1))
      except Exception as e:
        print(e)
        raise e
  cv2.imshow('MediaPipe Face Mesh', image)
  if (cv2.waitKey(5) & 0xFF == 27) or (cv2.waitKey(1) & 0xFF == ord('q')):
    break

  time.sleep(1)
  n += 1

# 释放摄像头资源并关闭窗口
cap.release()
cv2.destroyAllWindows()