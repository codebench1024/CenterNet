import gdal
import gdalconst
import os
import cv2
import numpy as np
import math

GDAL_ALLOW_LARGE_LIBJPEG_MEM_ALLOC=1048576000

def norm_image1(ds_array):
    # 将gdal读入的图像矩阵（16位，像素值0-65535，一般分布在低值像素），进行均衡化，放缩到0-255之间
    ds_array[ds_array >= 255] = 254    # 超过255的，直接取254
    umean, uvar, ustd = np.mean(ds_array), np.var(ds_array), np.std(ds_array)
    print(umean, uvar, ustd, np.max(ds_array), np.min(ds_array), np.size(ds_array))
    return ds_array

def norm_image2(ds_array, gap=10, over_rate = 0.995):
    # 将gdal读入的图像矩阵（16位，像素值0-65535，一般分布在低值像素），进行均衡化，放缩到0-255之间
    # 采用  （像素值-最小值）/（最大值-最小值）的归一化方式，然后放大到0-255之间
    # gap: 每gap像素划分直方图。 over_rate： 直方图中超过该rate的像素舍弃
    im_counts, im_ths = np.histogram(ds_array, bins=list(range(0,65536,gap)))
    cum_ratios = np.cumsum(im_counts) * 1.0 / np.sum(im_counts)
    # 找到99%以上的像素值分布，然后超过该值的舍弃掉
    tmp_rate, cum_idx = 0, 0
    while tmp_rate <= over_rate and cum_idx < len(cum_ratios):
        tmp_rate = cum_ratios[cum_idx]
        cum_idx += 1
    min_num, max_num = 0, cum_idx * gap  # 确定最小值，最大值
    print(im_counts, im_ths)
    print(max_num)
    ds_array[ds_array >= max_num] = max_num
    max_num = max_num / 255

    ds_array = ds_array // max_num

    return ds_array

def cal_log(base, max_num):
    # a为底数，max_num为计算到的最大值
    log_dict = {}
    for i in range(1, max_num):
        log_dict[i] = int(math.log(i, base))
    return log_dict

def norm_image3(ds_array):
    # 将gdal读入的图像矩阵（16位，像素值0-65535，一般分布在低值像素），进行均衡化，放缩到0-255之间
    # 放缩方法 采用分段的log函数
    '''
    im_counts, im_ths = np.histogram(ds_array, bins=list(range(0,65536,10)))
    cum_ratios = np.cumsum(im_counts) * 1.0 / np.sum(im_counts)

    # 找到99%以上的像素值分布，然后超过该值的舍弃掉
    tmp_rate, cum_idx = 0, 0
    while tmp_rate <= 0.995 and cum_idx < len(cum_ratios):
        tmp_rate = cum_ratios[cum_idx]
        cum_idx += 1

    min_num, max_num = 0, cum_idx * 10  # 确定最小值，最大值
    '''
    umean, uvar, ustd = np.mean(ds_array), np.var(ds_array), np.std(ds_array)

    min_num, max_num = 0, umean + 5 * ustd

    # 分段函数，进行log变换（底数的计算方式：16位tiff中像素值为mean+5*std的，映射到8位png为200），超过255的以255计
    base = math.pow(max_num, 1/180)
    upper_bound = int(base**255)

    log_dict = cal_log(base, upper_bound)

    # 将匿名函数向量化，然后对每个值进行log映射，超出范围的直接当做255.
    ds_array = np.vectorize(lambda x:log_dict.get(x, 255))(ds_array)

    return ds_array



def tiff_png(file_path, png_dir_path=""):
    # 将tiff转成png（png最大像素为65535*65535，因此只能转换低于此像素的tiff）
    dataset = gdal.Open(file_path)
    ds_array = dataset.ReadAsArray()
    png_array = norm_image3(ds_array)
    if png_dir_path == "":
        dir_name, file_name = os.path.split(file_path)
        to_write_png_path = os.path.join(dir_name, "%s.png" % file_name.split(".")[0])
    else:
        to_write_png_path = os.path.join(png_dir_path, os.path.basename(file_path) + ".png")
    cv2.imwrite(to_write_png_path, png_array)


if __name__ == '__main__':
    file_path = 'E:\\IMAGE_VV_SRA_scan_007.tif'
    dataset = gdal.Open(file_path)
    ds_array = dataset.ReadAsArray()
    umean, uvar, ustd = np.mean(ds_array), np.var(ds_array), np.std(ds_array)
    print(umean, uvar, ustd, np.max(ds_array), np.min(ds_array), np.size(ds_array))
    tiff_png(file_path)