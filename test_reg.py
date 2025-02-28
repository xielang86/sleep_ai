import cv2
from algorithm.pose_detector import PoseDetector,PoseResult
import os,sys,json,string
from enum import Enum

from dataclasses import asdict, dataclass,field

@dataclass
class Result:
  image_path: string = "",
  pose_info: PoseResult = field(default_factory = PoseResult),
  def __str__(self):
    return json.dumps(asdict(self), indent=2, default=str)


class CustomEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, Enum): 
      return obj.name
    return asdict(obj)

# input: sleep result, json
# output: result for regression test
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
def DetectAllImage(file_root, output_file):
  print(f"trace for {file_root}, {output_file}")
  pose_detector = PoseDetector()
  results = []
  # 遍历指定目录
  for root, dirs, files in os.walk(file_root):
    for file in files:
      # 获取文件扩展名并转换为小写
      file_extension = os.path.splitext(file)[1].lower()
      # 检查文件扩展名是否为支持的图片格式
      if file_extension not in image_extensions:
        continue
      # 拼接文件的完整路径
      file_path = os.path.join(root, file)
      # 使用cv2.imread读取图片
      image = cv2.imread(file_path)
      if image is None:
        sys.stderr.write(f"erro read image{file_path}")
        continue
      pose_result = pose_detector.Detect(0, image)
      if pose_result == None:
        sys.stderr.write(f"failed to detect {file_path}\n")
        continue
      r = Result()
      r.image_path = file_path
      r.pose_info = pose_result
      results.append(r)

# 使用 with 语句打开文件并写入数据
  with open(output_file, 'w', encoding='utf-8') as file:
  # 使用 json.dump() 将列表写入文件
    json.dump(results, file, ensure_ascii=False, indent=2, cls=CustomEncoder)

class StatIdx(Enum):
  TruePos = 0
  TrueNeg = 1
  FalsePos = 2
  FalseNeg = 3

def CalcStat(ground_truth, result, key, pos_value, stats, image_path, bad_case):
  result_value = result[key]
  ground_value = ground_truth[key]
  is_err = False
  if result_value == ground_value:
    if result_value == pos_value:
      stats[StatIdx.TruePos.value] += 1
    else:
      stats[StatIdx.TrueNeg.value] += 1
  else:
    is_err = True
    if result_value == pos_value:
      stats[StatIdx.FalsePos.value] += 1
    else:
      stats[StatIdx.FalseNeg.value] += 1

  if is_err and bad_case.get(image_path) is None:
    bad_case[image_path] = (ground_truth, result)

def CalcPR(stats):
  precision = 1.0 * stats[StatIdx.TruePos.value] / (stats[StatIdx.FalsePos.value]+ stats[StatIdx.TruePos.value])
  recall = 1.0 * stats[StatIdx.TruePos.value] / (stats[StatIdx.FalseNeg.value]+ stats[StatIdx.TruePos.value])
  return (precision, recall)

def DoRegressionTest(regression_file, err_file):
  pose_detector = PoseDetector()
  # inter pose would be as positive
  body_stat = [0] * 4
  left_hand_stat = [0] * 4
  right_hand_stat = [0] * 4
  left_eye_stat = [0] * 4
  right_eye_stat = [0] * 4
  face_direct = [0] * 4
  
  fout = open(err_file, "w")  
  bad_case = {}
  with open(regression_file, 'r') as file:
    data = json.load(file) 
    for record in data:
      ground_truth = record["pose_info"]
      image_path = record["image_path"]
      image = cv2.imread(image_path)
      if image is None:
        sys.stderr.write(f"erro read image{image_path}")
        continue
      
      try:
        pose_result = pose_detector.Detect(0, image)
        result = json.loads(str(pose_result))
      except Exception as e:
        print(e)
        raise e

      # compare pose_result and ground_truth
      try:
        CalcStat(ground_truth, result, "body", "SitDown", body_stat, image_path, bad_case)
      except Exception as e:
        print(e)

      CalcStat(ground_truth, result, "left_hand", "LiftOn", left_hand_stat, image_path, bad_case)
      CalcStat(ground_truth, result, "right_hand", "LiftOn", right_hand_stat, image_path, bad_case)
      CalcStat(ground_truth, result, "left_eye", "Open", left_eye_stat, image_path, bad_case)
      CalcStat(ground_truth, result, "right_eye", "Open", right_eye_stat, image_path, bad_case)

  print(body_stat)
  print(left_hand_stat)
  print(right_hand_stat)
  print(left_eye_stat)
  print(right_eye_stat)

  # print(bad_case.items())
  for cur_path, value_pair in bad_case.items():
    try:
      fout.write(cur_path) 
      fout.write("\n")
      fout.write(str(value_pair[0]))
      fout.write("\n")
      fout.write(str(value_pair[1]))
      fout.write("\n")
    except Exception as e:
      print(e)
      raise e

  fout.close()

  # calc precsion and recall
  print(CalcPR(body_stat))
  print(CalcPR(left_hand_stat))
  print(CalcPR(right_hand_stat))
  print(CalcPR(left_eye_stat))
  print(CalcPR(right_eye_stat))
   
def main():
  # 读取图片
  test_file = sys.argv[1]
  # single line means a root for all image, to generate the images result 
  lines = open(test_file).readlines()
  if len(lines) == 1:
    input_dir,output_file=lines[0].split(",")
    DetectAllImage(input_dir, output_file)
  else:
    DoRegressionTest(test_file, sys.argv[2])

if __name__ == "__main__":
  main()