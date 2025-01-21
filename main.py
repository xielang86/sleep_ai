import time
from algorithm import pose
import mediapipe as mp
import cv2,os

# 初始化mediapipe的手部解决方案
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# 打开摄像头
cap = cv2.VideoCapture(0)

detector = pose.PoseDetector()
n = 0
while cap.isOpened() and n < 3:
  ret, frame = cap.read()
  if not ret:
    print("read failed!")
    break
  # 将图像从BGR格式转换为RGB格式（mediapipe要求的输入格式）
  image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  try: 
    result = detector.Detect(image_rgb)
    if result == None:
      print("failed to detect")
    else:
      print(result.leftEyeClosed, result.rightEyeClosed)
      print(result.pose_type.name)
  except Exception as e:
    # 捕捉任何异常并抛出
    print(e)
    raise e

    # 显示处理后的图像
  cv2.imshow('Hand Detection', image_rgb)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
  time.sleep(1)
  n += 1

# 释放摄像头资源并关闭窗口
cap.release()
cv2.destroyAllWindows()