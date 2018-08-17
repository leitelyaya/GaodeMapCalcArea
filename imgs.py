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
import math

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
    bg_name = "data/B0FFF2CO42_1.png"
    img = io.imread(bg_name, as_grey=True)

    # 检测canny边缘,得到二值图片
    edgs = feature.canny(img, sigma=3)

    chull = morphology.convex_hull_object(edgs)

    fig, axes = plt.subplots(1, 2, figsize=(8, 8))
    ax0, ax1 = axes.ravel()
    ax0.imshow(edgs, plt.cm.gray)
    ax0.set_title('many objects')
    ax1.imshow(chull, plt.cm.gray)
    ax1.set_title('convex_hull image')
    plt.show()

def labels():
    bg_name = "data2/B0FFFF6TJG_1.png"
    data = io.imread(bg_name)
    data_rgb = color.rgba2rgb(data)
    data_gray = color.rgb2gray(data_rgb) # drop transparent layer

    # crop image
    mask = ~(data_gray == 1)
    mask_points = np.argwhere(mask)
    top,left,bottom,right = np.min(mask_points[:,0]), np.min(mask_points[:,1]), \
        np.max(mask_points[:,0]), np.max(mask_points[:,1])
    
    top = top-10>0 and top-10 or 0
    left = left-10>0 and left-10 or 0
    bottom = bottom+10<data.shape[0] and bottom+10 or data.shape[0]
    right = right+10<data.shape[1] and right+10 or data.shape[1]
        
    print("top,left,bottom,right", top,left,bottom,right);
        
    image = data_gray[top:bottom, left:right].copy()
    mask = mask[top:bottom, left:right]
    # print(image[0])
    
    # (np.isclose(image,0.0)) |
    """
    mask = ((np.isclose(image,0.928392)) | (np.isclose(image,0.844302)) | 
            (np.isclose(image,0.872443)) | (np.isclose(image,0.976991)) |
            (np.isclose(image,0.868027)) | (np.isclose(image,0.868027)) |
            (np.isclose(image,0.901961)) | (np.isclose(image,0.899898)) | 
            (np.isclose(image,0.884697)) |
            (np.isclose(image,0.901674)) | (np.isclose(image,0.8058))
            )
    """
    #mask = ~(image == 1)
    image[mask] = 0
    image[~mask] = 1    
    
    #edgs = feature.canny(image, sigma=3)

    #labels = measure.label(edgs, connectivity=1)  # 8连通区域标记
    #dst = color.label2rgb(labels)  # 根据不同的标记显示不同的颜色
    #print('regions number:', labels.max() + 1)  # 显示连通区域块数(从0开始标记)

    #dst = morphology.convex_hull_object(edgs)
    
    contours = measure.find_contours(image, 0.5)
    #cords = np.concatenate(contours)
    
    cordarr = []
    for cords in contours:
        appr_img = measure.subdivide_polygon(cords, degree=2, preserve_ends=True)
        appr_img = measure.approximate_polygon(appr_img, tolerance=1)
        appr_img += np.array([top,left])
        cordarr.append(appr_img.tolist())
        print("appr_img:", len(appr_img.tolist()))
    
    print("cordarr:", len(cordarr))
    

    f, (ax0, ax1) = plt.subplots(2, figsize=(15, 10))
    ax0.imshow(data)
    ax0.set_title('Input image')
    ax1.imshow(image)
    ax1.set_title('After mask')
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
    labels()
    # trim_data()