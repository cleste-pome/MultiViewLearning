import shutil
import os
import logging


def close_loggers():
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        handler.close()


def move_folder_contents(src, dst):
    """
    将源文件夹下的所有内容（包括文件和子文件夹）移动到目标文件夹。
    如果目标文件夹不存在，则创建它。移动完成后源文件夹会变为空文件夹。
    :param src: 要移动内容的源文件夹路径
    :param dst: 目标文件夹路径
    """
    # 检查源文件夹是否存在
    if not os.path.exists(src):
        print(f"源文件夹 {src} 不存在！")
        return

    # 如果目标文件夹不存在，则创建它
    if not os.path.exists(dst):
        try:
            os.makedirs(dst)
            print(f"目标文件夹 {dst} 不存在，已创建。")
        except Exception as e:
            print(f"创建目标文件夹时出错: {e}")
            return

    try:
        # 遍历源文件夹中的所有文件和子文件夹
        for item in os.listdir(src):
            src_item = os.path.join(src, item)
            dst_item = os.path.join(dst, item)

            # 移动文件或文件夹
            shutil.move(src_item, dst_item)
            print(f"{src_item} 已成功移动到 {dst_item}")

        # 检查源文件夹是否为空
        if not os.listdir(src):
            print(f"源文件夹 {src} 已为空。")

    except Exception as e:
        print(f"移动文件夹内容时出错: {e}")


def move(missratio):
    src_folders = ['1.logs', '2.results_imgs', '3.csv']  # 源文件夹路径
    for src_folder in src_folders:
        dst_folder = f'{missratio}/{src_folder}'  # 目标文件夹路径
        # 在移动文件前调用
        close_loggers()
        move_folder_contents(src_folder, dst_folder)
