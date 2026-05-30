

import os

def get_project_root() -> str:
    """
    获取工程根目录
    """

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    return project_root



def get_abs_path(relative_path: str) -> str:
    """
    获取相对于工程根目录的绝对路径
    """
    project_root = get_project_root()
    abs_path = os.path.join(project_root, relative_path)
    return abs_path


if __name__ == "__main__":
    xx=get_abs_path(relative_path='config/zzz.txt')
    print(xx)
