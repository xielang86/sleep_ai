import logging
def CreateCustomLogger(log_filename, code_filename, log_level):
  # 创建一个 Logger 对象
  logger = logging.getLogger(code_filename)
  # 设置日志级别为 DEBUG，这样可以记录所有级别的日志信息
  logger.setLevel(logging.DEBUG)

  # 创建一个文件处理器，将日志信息写入指定的文件
  file_handler = logging.FileHandler(log_filename)
  # 设置文件处理器的日志级别为 DEBUG
  # file_handler.setLevel(logging.DEBUG)
  file_handler.setLevel(log_level)

  # 定义日志格式
  formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
  # 将日志格式应用到文件处理器
  file_handler.setFormatter(formatter)

  # 将文件处理器添加到 Logger 对象中
  logger.addHandler(file_handler)
  return logger