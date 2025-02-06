import cv2
from sleep_phrase import SleepPhraseDetector
import math,sys

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
    print(result)

if __name__ == "__main__":
  main()