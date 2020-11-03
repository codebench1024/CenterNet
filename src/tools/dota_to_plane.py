import os
import sys


def dota_to_plane(before_dir, after_dir):
    # 将merge之后的dota格式的txt结果：文件名为类型，第一列为图片名称
    # 转成plane格式的：也就是比赛时的格式: label confidence x1 y1 x2 y2
    filenames = os.listdir(before_dir)
    for filename in filenames:
        if filename[-3:] != "txt":
            continue
        with open(os.path.join(before_dir, filename), 'r') as file1:
            lines = file1.readlines()
            for line in lines:
                line = line.strip().split()
                if len(line) < 5:
                    continue
                to_write_name = line[0] + ".txt"
                to_write_name = os.path.join(after_dir, to_write_name)
                with open(to_write_name, 'a') as file2:
                    class_name = filename[6:-4]
                    file2.write("%s %s %s %s %s %s\n" % (class_name, line[1], line[2], line[3], line[4], line[5]))


if __name__ == '__main__':
    before_dir = "E:\\遥感图像稀疏表征与智能分析大赛\\tmp1"
    after_dir = "E:\\遥感图像稀疏表征与智能分析大赛\\tmp2"
    dota_to_plane(before_dir, after_dir)