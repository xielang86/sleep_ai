import sys,math
import numpy as np

def show_file_and_line(frame)-> str:
  file_name = frame.f_code.co_filename
  line_number = frame.f_lineno
  return f"{file_name}:{line_number}"

def distance(a, b)->float :
  x = a.x - b.x
  y = a.y - b.y
  return math.sqrt(x * x + y * y)

def distance_pair(a, b)->float:
  x = a[0] - b[0]
  y = a[1] - b[1]
  return math.sqrt(x*x + y*y)

def ChangeToVector(obj, flds):
  float_values = []
  for field in flds:
    value = getattr(obj, field)
    if isinstance(value, float) and not isinstance(value, bool):  # 排除 bool（bool 是 float 子类）
      float_values.append(value)
  
  return np.array(float_values, dtype=np.float32)  # 转换为浮点向量

def CalculateRelativeDiff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
  """
  (a - b) / a
  """
  if a.shape != b.shape:
    raise ValueError("shape must be same")
  
  mask = a == 0 # 找到 a 中为 0 的位置
  c = np.where (mask, b, (a - b) /a) # 条件赋值
  return c
  
def NormAngle(angle):
  if angle < -180 or angle > 180:
    return -1

  if angle < 0:
    angle = 0 - angle

  if angle > 90:
    angle = 180 - angle
  
  return angle

def CalculateThreePointAngle(a, b, c):
  """
  计算由三个点 a, b, c 形成的角度
  """
  radians = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
  angle = math.degrees(radians)
  angle = abs(angle)
  if angle > 180:
      angle = 360 - angle
  return angle

