from cache import *
import time

# 示例使用
cache = ThreadSafeCache(max_size=3, default_expiration=5)

def worker_get(cache, key):
  result = cache.get(key)
  print(f"Get {key}: {result}")

def worker_set(cache, key, value):
  cache.set(key, value)
  print(f"Set {key}: {value}")

def TestCache():
  threads = []
  for i in range(5):
    t1 = threading.Thread(target=worker_set, args=(cache, f"key{i}", f"value{i}"))
    t2 = threading.Thread(target=worker_get, args=(cache, f"key{i}"))
    threads.extend([t1, t2])
    t1.start()
    t2.start()

  for t in threads:
    t.join()

if __name__ == "__main__":
  TestCache()