import cv2
from sleep_phrase import SleepPhraseDetector
import math,sys

def calculate_head_angle(nose_landmark, neck_landmark):
  # 计算头部仰起角度（弧度）
  dx = nose_landmark.x - neck_landmark.x
  dy = nose_landmark.y - neck_landmark.y
  angle_rad = math.atan2(dy, dx)
  # 转换为角度
  angle_deg = math.degrees(angle_rad)
  return angle_deg


def main():
  # 读取图片
  image_paths = sys.argv[1].split(",")

  detector = SleepPhraseDetector(2)
  images = []
  for image_path in image_paths: 
    image = cv2.imread(image_path)
    if image is None:
      print(f"Could not read the image at {image_path}")
      continue
    # 将 BGR 图像转换为 RGB
    images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
  result = detector.DetectSleepPhrase(0, 0, images, None)
  if result == None:
    print("failed to detect")
  else:
    print(result.to_json())

if __name__ == "__main__":
  main()