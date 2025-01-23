import sys
def show_file_and_line(frame)-> str:
  file_name = frame.f_code.co_filename
  line_number = frame.f_lineno
  return f"{file_name}:{line_number}"