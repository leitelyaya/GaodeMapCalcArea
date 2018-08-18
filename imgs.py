import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
from skimage import io, morphology, feature, filters, img_as_float
from skimage import measure, color
from skimage import exposure
from skimage import data
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from skimage.transform import warp, AffineTransform
import numpy as np
import codecs
import json
import os

def complex_images():
    bg_name = "data/B0FFF2CO42_1.png"
    fname = "data/B0FFF2CO42_GD0003800210101072.png"
    fname2 = "data/c_B0FFF2CO42_GD0003800210101072.png"

    bg = Image.open(bg_name)
    bg = bg.convert('L').convert('RGBA')
    mask = Image.new("RGBA", bg.size)
    final = Image.new("RGBA", bg.size)

    source = Image.open(fname)
    mask.paste(source, (546, 159))

    final = Image.alpha_composite(final, bg)
    final = Image.alpha_composite(final, mask)

    final.save(fname2)


def split_space_content():
    for root, dirs, files in os.walk('data'):
        for fname in files:
            bg_name = "data/"+fname
            img = io.imread(bg_name, as_grey=True)

            # 检测canny边缘,得到二值图片
            edgs = feature.canny(img, sigma=3)

            chull = morphology.convex_hull_object(edgs)

            #
            # fig, axes = plt.subplots(1, 2, figsize=(8, 8))
            # ax0, ax1 = axes.ravel()
            # ax0.imshow(edgs, plt.cm.gray)
            # ax0.set_title('many objects')
            # ax1.imshow(chull, plt.cm.gray)
            # ax1.set_title('convex_hull image')
            # # plt.show()

            plt.imsave("data2/"+fname, chull)

def to_contours():
    for root, dirs, files in os.walk('data2'):
        for fname in files:
            dst = io.imread('data2/'+fname, as_grey=True)
            contours = measure.find_contours(dst, 0.5)
            cords = np.concatenate(contours)

            new_img = measure.subdivide_polygon(cords, degree=2, preserve_ends=True)
            appr_img = measure.approximate_polygon(new_img, tolerance=1)

            print(fname, len(appr_img.tolist()))

def labels():
    bg_name = "data/test.png"
    data = io.imread(bg_name)
    # data = color.rgba2rgb(data)

    # edgs = feature.canny(data, sigma=3)

    # labels = measure.label(edgs, connectivity=1)  # 8连通区域标记
    # dst = color.label2rgb(labels)  # 根据不同的标记显示不同的颜色
    # print('regions number:', labels.max() + 1)  # 显示连通区域块数(从0开始标记)

    image = data.copy()
    # print(image[0])

    image[image == (204, 227, 232)] = 0

    f, (ax0, ax1) = plt.subplots(2, figsize=(15, 10))
    ax0.imshow(data)
    ax0.set_title('Input image')
    ax1.imshow(image)
    ax1.set_title('Marker locations')
    ax1.plot()
    ax1.axis('image')
    plt.show()

def trim_data():
    suzu = codecs.open("extsource/poi_suzu_mall_names.txt", encoding="gbk")
    names = codecs.open("extsource/poi_mall_names.txt", encoding="gbk")

    am = set()
    for line in suzu:
        if line:
            s = json.loads(line)
            am.add(s["id"])

    for line in names:
        if line:
            s = json.loads(line)
            if s["id"] in am:
                am.remove(s["id"])

    print("\r\n".join(am))


if __name__ == '__main__':
    # complex_images()
    # split_space_content()
    to_contours()
    # labels()
    # trim_data()