import math
def distance(a, b)->float :
  x = a.x - b.x
  y = a.y - b.y
  return math.sqrt(x * x + y * y)

def distance_pair(a, b)->float:
  x = a[0] - b[0]
  y = a[1] - b[1]
  return math.sqrt(x*x + y*y)