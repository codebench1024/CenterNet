import os
import sys


def groundtruth_plane_to_dota(source_dir, dest_dir):
    '''
    source_dir -- SAR plane的groundtruth: label x1 y1 x2 y2
    dest_dir   -- dota的groundtruth: x1 y1 x2 y1 x2 y2 x1 y2 label is_difficult
    为了统一格式，于是将SAR plane的格式先转成dota的格式
    '''
    filenames = os.listdir(source_dir)
    for filename in filenames:
        with open(os.path.join(source_dir, filename), 'r') as file1:
            lines = file1.readlines()
            with open(os.path.join(dest_dir, filename), 'w') as file2:
                for line in lines:
                    line = line.strip().split()
                    if len(line) < 4:
                        continue
                    file2.write("%s %s %s %s %s %s %s %s %s 1" % (line[1], line[2], line[3], line[2], line[3], line[4], line[1], line[4], line[0]))

def ground_dota_to_plane(source_dir, dest_dir):
    '''
    source_dir -- dota的groundtruth: x1 y1 x2 y1 x2 y2 x1 y2 label is_difficult
    dest_dir   -- SAR plane的groundtruth: label x1 y1 x2 y2
    为了统一格式，最初是将5列转成10列便于调用脚本切割图像，切割完成后，DOTA2COCO
    '''


