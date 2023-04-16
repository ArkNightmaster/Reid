import os


def mkdir_if_missing(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        return True
    return False