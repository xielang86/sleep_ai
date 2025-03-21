import cv2
from algorithm.feature_extractor import FeatureExtractor,PoseFeature
import os,sys,json,string
from enum import Enum

import mediapipe as mp
from dataclasses import asdict, dataclass,field

def Extract(data_file, fea_file):
  mp_pose = mp.solutions.pose
  extractor = FeatureExtractor()
  pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.03,
                                 min_tracking_confidence=0.01)

  face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.05, min_tracking_confidence=0.1)

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
      
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      mp_result = pose.process(image)
      if mp_result is None or mp_result.pose_landmarks is None:
        print("mediapipe detect none body")
        continue

      landmarks = mp_result.pose_landmarks
      face_results = face_mesh.process(image)
      try:
        ih, iw, _ = image.shape
        fea = extractor.Extract(landmarks, face_results, ih, iw)
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