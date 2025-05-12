import cv2
from sleep_phrase import SleepPhraseDetector
from algorithm import pose,pose_detector
import mediapipe as mp
import sys
# 定义不同部位的颜色
HEAD_COLOR = (0, 255, 0)  # 绿色
TORSO_COLOR = (255, 0, 0)  # 蓝色
LEGS_COLOR = (0, 0, 255)  # 红色
HANDS_COLOR = (255, 255, 0)  # 青色

# 定义不同部位的关键点索引
HEAD_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
TORSO_INDICES = [11, 12, 13, 14, 15, 16, 23, 24]
LEGS_INDICES = [25, 26, 27, 28, 29, 30, 31, 32]
HANDS_INDICES = [15, 16, 17, 18, 19, 20, 21, 22]
# 定义不同部位的连接关系
HEAD_CONNECTIONS = [(0, 1), (0, 4), (1, 2), (2, 3), (3, 7), (4, 5), (5, 6), (6, 8), (9, 10)]
TORSO_CONNECTIONS = [(11, 12), (11, 13), (12, 14), (13, 15), (14, 16), (11, 23), (12, 24), (23, 24)]
LEGS_CONNECTIONS = [(23, 25), (24, 26), (25, 27), (26, 28), (27, 29), (28, 30), (29, 31), (30, 32)]
HANDS_CONNECTIONS = [(15, 17), (17, 19), (19, 21), (16, 18), (18, 20), (20, 22)]

def draw_landmarks_with_colors(image, results):
  if results.pose_landmarks:
    h, w, c = image.shape
    # 绘制头部关键点及连线
    for idx in HEAD_INDICES:
      landmark = results.pose_landmarks.landmark[idx]
      x = int(landmark.x * w)
      y = int(landmark.y * h)
      cv2.circle(image, (x, y), 5, HEAD_COLOR, -1)
    for connection in HEAD_CONNECTIONS:
      start_idx, end_idx = connection
      if start_idx in HEAD_INDICES and end_idx in HEAD_INDICES:
        start_landmark = results.pose_landmarks.landmark[start_idx]
        end_landmark = results.pose_landmarks.landmark[end_idx]
        start_x = int(start_landmark.x * w)
        start_y = int(start_landmark.y * h)
        end_x = int(end_landmark.x * w)
        end_y = int(end_landmark.y * h)
        cv2.line(image, (start_x, start_y), (end_x, end_y), HEAD_COLOR, 2)

    # 绘制躯干关键点及连线
    for idx in TORSO_INDICES:
      landmark = results.pose_landmarks.landmark[idx]
      x = int(landmark.x * w)
      y = int(landmark.y * h)
      cv2.circle(image, (x, y), 5, TORSO_COLOR, -1)
    for connection in TORSO_CONNECTIONS:
      start_idx, end_idx = connection
      if start_idx in TORSO_INDICES and end_idx in TORSO_INDICES:
        start_landmark = results.pose_landmarks.landmark[start_idx]
        end_landmark = results.pose_landmarks.landmark[end_idx]
        start_x = int(start_landmark.x * w)
        start_y = int(start_landmark.y * h)
        end_x = int(end_landmark.x * w)
        end_y = int(end_landmark.y * h)
        cv2.line(image, (start_x, start_y), (end_x, end_y), TORSO_COLOR, 2)

    # 绘制腿部关键点及连线
    for idx in LEGS_INDICES:
      landmark = results.pose_landmarks.landmark[idx]
      x = int(landmark.x * w)
      y = int(landmark.y * h)
      cv2.circle(image, (x, y), 5, LEGS_COLOR, -1)
    for connection in LEGS_CONNECTIONS:
      start_idx, end_idx = connection
      if start_idx in LEGS_INDICES and end_idx in LEGS_INDICES:
        start_landmark = results.pose_landmarks.landmark[start_idx]
        end_landmark = results.pose_landmarks.landmark[end_idx]
        start_x = int(start_landmark.x * w)
        start_y = int(start_landmark.y * h)
        end_x = int(end_landmark.x * w)
        end_y = int(end_landmark.y * h)
        cv2.line(image, (start_x, start_y), (end_x, end_y), LEGS_COLOR, 2)

    # 绘制手部关键点及连线
    for idx in HANDS_INDICES:
      landmark = results.pose_landmarks.landmark[idx]
      x = int(landmark.x * w)
      y = int(landmark.y * h)
      cv2.circle(image, (x, y), 5, HANDS_COLOR, -1)
    for connection in HANDS_CONNECTIONS:
      start_idx, end_idx = connection
      if start_idx in HANDS_INDICES and end_idx in HANDS_INDICES:
        start_landmark = results.pose_landmarks.landmark[start_idx]
        end_landmark = results.pose_landmarks.landmark[end_idx]
        start_x = int(start_landmark.x * w)
        start_y = int(start_landmark.y * h)
        end_x = int(end_landmark.x * w)
        end_y = int(end_landmark.y * h)
        cv2.line(image, (start_x, start_y), (end_x, end_y), HANDS_COLOR, 2)
  return image

def main():
  # 读取图片
  image_paths = sys.argv[1].split(",")
  mp_pose = mp.solutions.pose

  detector = pose_detector.PoseDetector( \
      mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, min_tracking_confidence=0.1), \
      mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.05, min_tracking_confidence=0.1))

  mp_drawing = mp.solutions.drawing_utils
  for image_path in image_paths: 
    image = cv2.imread(image_path)
    if image is None:
      print(f"Could not read the image at {image_path}")
      continue
    result = detector.Detect(0, image)
    # 释放资源
    try:
      mp_result = detector.get_mp_result()
      if mp_result.pose_landmarks:
      # 在图片上绘制关键点和连接线
        mp_drawing.draw_landmarks(
          image,
          mp_result.pose_landmarks,
          mp_pose.POSE_CONNECTIONS,
          mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
          mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
        )
    except Exception as e:
      print(e)
      raise e
    image_with_landmarks = draw_landmarks_with_colors(image, mp_result)

    if result == None:
      print("failed to detect")
    else:
      print(result)

    cv2.imwrite('output2.jpg', image_with_landmarks)

    cv2.imwrite('output.jpg', image)

if __name__ == "__main__":
  main()