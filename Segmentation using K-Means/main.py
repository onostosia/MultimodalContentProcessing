import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import collections
import cv2
import os
import re
from tqdm import tqdm

def load_image(name, output=False):
    img = cv2.imread(path + '/' + name + '.png')
    # img = cv2.cvtColor(img ,cv2.COLOR_BGR2RGB)
    img = cv2.normalize(img, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return img

def load_mask(name, output=False):
    img = cv2.imread(path + '/' + name + 'm.png', cv2.IMREAD_GRAYSCALE)
    return np.array(img).reshape(img.shape[0], img.shape[1], 1)


def show_img_by_path(display_list):
    plt.figure(figsize=(15, 15))
    title = ['Input Image', 'True Mask', 'Predicted Mask']
    for i in range(len(display_list)):
        img_path = path + '/' + display_list[i] + '.png'
        image = plt.imread(img_path)
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(image)
        plt.axis('off')
    plt.show()

def display(display_list, title=['Input Image', 'True Mask', 'Predicted Mask']):
    plt.figure(figsize=(15, 15))
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(display_list[i])
        plt.axis('off')
    plt.show()

def blur_img(img, blur_param = 30, center = (100,100), radius = 60):
    blurred_img = cv2.GaussianBlur(img, (21, 21), blur_param)
    mask = np.zeros(img.shape, dtype=np.uint8)
    mask = cv2.circle(mask, center, radius, (255, 255, 255), -1)
    out = np.where(mask==np.array([255, 255, 255]), img, blurred_img)
    return out

def print_cluster(result, cluster_tag):
    result_copy = result.copy()
    result_copy[abs(result_copy - cluster_tag) < 0.0001] = 1
    result_copy[result_copy != 1] = 0
    return result_copy

def get_IoU_for_slice(slice_, mask):
    intersection = np.logical_and(slice_, mask)
    union = np.logical_or(slice_, mask)
    iou_score = np.sum(intersection)/ np.sum(union)
    return iou_score

def get_IoU(slices):
    return np.array(slices).max(), np.argmax(np.array(slices))

def accuracy_IoU_scorer(result, mask):
    metrics_iou = []
    for i in np.unique(result.flatten()):
        result_copy = result.copy()
        result_copy[abs(result_copy - i) < 0.0001] = 1
        result_copy[result_copy != 1] = 0
        metrics_iou.append(get_IoU_for_slice(result_copy, mask))

    accuracy, cluster_ind = get_IoU(metrics_iou)
    #print('---', cluster_ind, np.unique(result.flatten())[cluster_ind])
    slice_ = print_cluster(result, np.unique(result.flatten())[cluster_ind])
    return accuracy, slice_


def KMeans_img_segment(img, K=6):
    vectorized_img = np.float32(img.reshape((-1, 3)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    attempts = 10

    ret, label, center = cv2.kmeans(vectorized_img,
                                    K,
                                    None,
                                    criteria,
                                    attempts,
                                    cv2.KMEANS_PP_CENTERS
                                    )
    res = center[label.flatten()]
    result_image = res.reshape((img.shape))
    return result_image, label, center


def get_optimal_k(img, mask, k_range=range(6, 19)):
    iou_scores = []
    for k in k_range:
        accuracy_list = []
        for _ in range(10):
            result, res_labels, res_center = KMeans_img_segment(img, k)
            accuracy, slice_ = accuracy_IoU_scorer(result, mask)
            accuracy_list.append(accuracy)
        iou_scores.append(np.array(accuracy_list).mean())
    return k_range[np.argmax(iou_scores)]

def save_accuracy_in_file():
    return ''

def save_predicted_mask_in_file():
    return ''

def load_img_from_file(file_name):
    data_2d = np.loadtxt(file_name, delimiter=',')
    # reshape the 2D array to a 3D array with shape (200, 200, 3)
    data_3d = data_2d.reshape((200, 200, -1))
    return data_3d

def get_unprocessed_images(processed_images_path, all_images_names):
    Lines = open(processed_images_path, 'r').readlines()
    uniq_processed_images = []
    for line in Lines:
        uniq_processed_images.append(line.strip())

    return list(set(all_images_names) - set(uniq_processed_images))



processed_images_path_local = '/Users/anastasia/PycharmProjects/pythonProject2/processed_images.txt'
path_local = '/Users/anastasia/Desktop/Datasets/Total/BlackPupilEye'

path = '/home/nastya/dataset/test/Total/BlackPupilEye'
processed_images_path = '/home/nastya/python/processed_images_without_blur.txt'

image_path_list = os.listdir(path)
image_names = [re.sub('m.png|.png', '', img) for img in image_path_list]
uniq_img_names = list(set(image_names))

uniq_img_names = get_unprocessed_images(processed_images_path, uniq_img_names)
n = 100
images = np.array([load_image(img) for img in (uniq_img_names[:n])])
masks = np.array([load_mask(img) for img in (uniq_img_names[:n])])

for i in range(n):
    img = images[i] 
    #img = blur_img(images[i])

    optimal_k = get_optimal_k(img, masks[i])

    result, res_labels, res_center = KMeans_img_segment(img, optimal_k)
    accuracy, slice_ = accuracy_IoU_scorer(result, masks[i])

    with open('out/imagesWithoutBlur/metricsKMeans.txt', "a") as f:
        f.write(uniq_img_names[i] + ' ' + str(accuracy) + ' ' + str(optimal_k) + '\n')
        f.close()

    data_2d = slice_.reshape((-1, slice_.shape[-1]))
    np.savetxt('out/imagesWithoutBlur/imagesKMeans' + (uniq_img_names[i]) + '.txt', data_2d, delimiter=',')


