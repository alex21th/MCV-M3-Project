import argparse
import os
import random
import shutil


def move_random_files(src_folder, dst_folder, num_files):
    """
    Move a specified number of randomly selected files from one folder to another.

    :param src_folder: The source folder that contains the files to be moved.
    :param dst_folder: The destination folder to which the files will be moved.
    :param num_files: The number of files to be moved.
    """
    files = os.listdir(src_folder)
    selected_files = random.sample(files, num_files)

    for file in selected_files:
        src_path = os.path.join(src_folder, file)
        dst_path = os.path.join(dst_folder, file)
        shutil.move(src_path, dst_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Move a specified number of randomly selected files from one folder to another.')
    parser.add_argument('src_folder', type=str, help='The source folder that contains the files to be moved.')
    parser.add_argument('dst_folder', type=str, help='The destination folder to which the files will be moved.')
    parser.add_argument('num_files', type=int, help='The number of files to be moved.')
    args = parser.parse_args()

    move_random_files(args.src_folder, args.dst_folder, args.num_files)
