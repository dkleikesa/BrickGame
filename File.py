import shutil
import os


def copy_file(src, dst):
    """
    复制文件从src到dst。
    如果dst是一个目录，文件将被复制到该目录下，保持原文件名。
    如果目标目录不存在，会自动创建目录。
    """
    # 确保源文件存在
    if not os.path.isfile(src):
        print(f"源文件 {src} 不存在.")
        return

    # 如果目标路径是目录，且目录不存在，则创建该目录
    if os.path.isdir(dst):
        os.makedirs(dst, exist_ok=True)
    else:
        # 如果目标路径不是目录，默认用户可能是希望在目标路径创建文件
        # 此时需要检查目标文件的目录部分是否存在，如果没有则创建
        parent_dir = os.path.dirname(dst)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)

    # 复制文件
    shutil.move(src, dst)
    print(f"文件已从 {src} 复制到 {dst}.")


if __name__ == '__main__':
    src_image_dir = '/Users/jianzhangli/src/BrickGame/classBitmap_data_origin'
    dest_image_dir = '/Users/jianzhangli/src/BrickGame/classBitmap_data'
    file_names = os.listdir(src_image_dir)
    for file_name in file_names:
        if not file_name.endswith('.png'):
            continue
        class_name = file_name.split('_')[0]
        full_src_path = os.path.join(src_image_dir, file_name)
        full_dest_path= os.path.join(dest_image_dir, class_name,file_name)
        copy_file(full_src_path,full_dest_path)