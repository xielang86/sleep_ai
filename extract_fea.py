import cv2
from algorithm.feature_extractor import FeatureExtractor,PoseFeature
import os,sys,json,string
from enum import Enum

import mediapipe as mp
from dataclasses import asdict, dataclass,field
    
def Extract(data_file, fea_file):
  mp_pose = mp.solutions.pose
  extractor = FeatureExtractor()

  field_names = list(PoseFeature.__dataclass_fields__.keys())
  header = '\t'.join(field_names)
  fout = open(fea_file, "w")  
  fout.write(header)

  with open(data_file, 'r') as file:
    data = json.load(file) 
    for record in data:
      ground_truth = record["pose_info"]
      image_path = record["image_path"]
      image = cv2.imread(image_path)
      if image is None:
        sys.stderr.write(f"erro read image{image_path}")
        continue
      
      try:
        fea = extractor.Extract(image)
        if fea == None:
          continue

        values = [str(getattr(fea, field)) for field in field_names]
        row = '\t'.join(values)
        fout.write("\n")
        fout.write(row)
        # pose_result = pose_detector.Detect(0, image)
        # result = json.loads((json.dumps(pose_result, cls=CustomEncoder)))
      except Exception as e:
        print(e)
        raise e
      # compare pose_result and ground_truth
  fout.close()

def main():
  # 读取图片
  data_file = sys.argv[1]
  sample_file = sys.argv[2]
  
  # single line means a root for all image, to generate the images result 
  Extract(data_file, sample_file)

if __name__ == "__main__":
  main()