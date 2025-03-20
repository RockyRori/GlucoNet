import sys
import os
import tkinter as tk

# 计算 `program/` 目录的绝对路径
PROGRAM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "program")

# 将 `program/` 目录添加到 Python 的 `sys.path`
sys.path.append(PROGRAM_DIR)

from local_app import DiabetesPredictionApp  # 现在可以正确导入


def run_local_app():
    """启动本地 GUI 应用"""
    try:
        root = tk.Tk()
        app = DiabetesPredictionApp(root)
        root.mainloop()
    except Exception as e:
        print(f"Error running local app: {e}")
        sys.exit(1)


if __name__ == "__main__":
    print("Starting Diabetes Prediction Local App...")
    run_local_app()
