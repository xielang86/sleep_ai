import math
def distance(a, b)->float :
  x = a.x - b.x
  y = a.y - b.y
  return math.sqrt(x * x + y * y)
