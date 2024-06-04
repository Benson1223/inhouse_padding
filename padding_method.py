


import os
import cv2
import numpy as np

def remove_background(img):
    mask = np.zeros(img.shape[:2], np.uint8)

    # 设置前景和背景模型
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # 定义矩形ROI，注意这里的矩形ROI应该包含所有前景物体
    rect = (50, 50, img.shape[1]-50, img.shape[0]-50)

    # 执行 GrabCut 算法
    cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    # 提取前景和可能的前景
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    result = img * mask2[:, :, np.newaxis]

    # 在原始背景部分填充检测到的物体周围的像素颜色
    result[mask2 == 0] = find_object_color(img, mask)

    return result

def find_object_color(img, mask):
    # 寻找物体的边界
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 计算物体的边界框
    x, y, w, h = cv2.boundingRect(contours[0])

    # 在物体周围取一个像素的区域，并返回区域内像素的平均颜色
    roi = img[y:y+h, x:x+w]
    color = np.mean(roi, axis=(0, 1)).astype(np.uint8)

    return color

def find_and_save(folder_path, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image = cv2.imread(os.path.join(folder_path, filename))
            # 去除背景
            result = remove_background(image)
            # 保存结果
            cv2.imwrite(os.path.join(save_path, filename), result)

folder_path = "./"
save_path = "./res"

find_and_save(folder_path, save_path)

print("Contour images saved successfully.")
