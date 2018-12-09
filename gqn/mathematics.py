import math
import numpy as np
import cv2


def yaw(eye, center):
    eye_x, eye_z = eye[0], eye[2]
    center_x, center_z = center[0], center[2]
    eye_direction = (center_x - eye_x, center_z - eye_z)
    frontward_direction = (0, 1)
    norm_eye = math.sqrt(eye_direction[0] * eye_direction[0] +
                         eye_direction[1] * eye_direction[1])
    cos = (eye_direction[0] * frontward_direction[0] +
           eye_direction[1] * frontward_direction[1]) / (norm_eye * 1.0)
    rad = math.acos(cos)
    if eye_direction[0] < 0:
        rad = -rad
    return rad


def pitch(eye, center):
    eye_direction = (center[0] - eye[0], center[1] - eye[1],
                     center[2] - eye[2])
    radius = math.sqrt(eye_direction[0]**2 + eye_direction[2]**2)
    rad = math.atan(eye_direction[1] / (radius + 1e-16))
    return rad


def get_KL_div(img_original, img_predict):
    assert img_original.shape == img_predict.shape
    assert len(img_original.shape) <= 3
    assert len(img_predict.shape) <= 3

    img_original = img_original.transpose((1, 2, 0)).astype(np.float32)
    img_predict = img_predict.transpose((1, 2, 0)).astype(np.float32)

    img_shape = img_original.shape # ex) (64, 64) expected for black and white image
    img_pixel_num = img_original.shape[0] * img_original.shape[1]

    if len(img_shape) == 3:
        img_original = cv2.cvtColor(img_original, cv2.COLOR_RGB2GRAY)
        img_predict = cv2.cvtColor(img_predict, cv2.COLOR_RGB2GRAY)


    img_original_flat = [int(item * 255) for sublist in img_original for item in sublist]
    img_predict_flat = [int(item * 255) for sublist in img_predict for item in sublist]

    bit_per_pixel = 8
    entropy_original = 0
    entropy_original_gen = 0

    for i in range(2**bit_per_pixel - 1):
        original_i_ratio = len([x for x in img_original_flat if x == i]) / img_pixel_num
        gen_i_ratio = len([x for x in img_predict_flat if x == i]) / img_pixel_num

        if gen_i_ratio > 0:
            entropy_original_gen += original_i_ratio * math.log(gen_i_ratio)
        if original_i_ratio > 0:
            entropy_original += original_i_ratio * math.log(original_i_ratio)

    entropy_original_gen = -entropy_original_gen
    entropy_original = -entropy_original

    KL_divergence = entropy_original_gen - entropy_original

    return KL_divergence if KL_divergence > 0 else 0

def get_squared_distance(img_original, img_predict):
    assert img_original.shape == img_predict.shape
    assert len(img_original.shape) <= 3
    assert len(img_predict.shape) <= 3

    total_distance = 0
    regional_distance = np.zeros((64, 64, 3))

    img_original = img_original.transpose((1, 2, 0)).astype(np.float32)
    img_predict = img_predict.transpose((1, 2, 0)).astype(np.float32)

    for i, (row_orig, row_pred) in enumerate(zip(img_original, img_predict)):
        for j, (dot_orig, dot_pred) in enumerate(zip(row_orig, row_pred)):
            for k, (rgb_orig, rgb_pred) in enumerate(zip(dot_orig, dot_pred)):
                distance = 0.5 * (rgb_orig - rgb_pred)**2
                total_distance += distance
                regional_distance[i][j][0] += distance
                regional_distance[i][j][1] += distance
                regional_distance[i][j][2] += distance

    regional_distance = regional_distance.transpose((2, 0, 1))

    return total_distance, regional_distance