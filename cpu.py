import numpy as np
import math

def change_brightness_cpu(image_array, amount):
    # convert to int16 to avoid overflow when adding
    output = image_array.astype(np.int16)

    for i in range(image_array.shape[0]): # height
        for j in range(image_array.shape[1]): # width
            for k in range(image_array.shape[2]): # color channel [r,g,b]

                # change the brightness of one color channel for one pixel
                new_value = output[i][j][k] + amount
                output[i][j][k] = np.clip(new_value, 0, 255)

    return output.astype(np.uint8)  # convert back to uint8 for image representation

def change_contrast_cpu(image_array, factor):
    factor = float(factor)
    contrast_scale = 1.0 + (factor / 100.0)
    output = image_array.astype(np.float32)

    for i in range(output.shape[0]):  # height
        for j in range(output.shape[1]):  # width
            for k in range(output.shape[2]):  # color channel [r,g,b]
                new_value = 128.0 + (output[i][j][k] - 128.0) * contrast_scale
                output[i][j][k] = np.clip(new_value, 0, 255)

    return output.astype(np.uint8)

def change_gamma_contrast_cpu(image_array, factor):
    factor = float(factor)
    output = image_array.astype(np.float32)

    for i in range(output.shape[0]):  # height
        for j in range(output.shape[1]):  # width
            for k in range(output.shape[2]):  # color channel [r,g,b]
                # normalizes the pixel value from 0-255 to 0-1
                normalized = output[i][j][k] / 255

                # the normalized pixel put into gamma correction formula
                normal_result = math.pow(normalized, factor)

                new_value = normal_result*255
                output[i][j][k] = np.clip(new_value, 0, 255)
    
    return output.astype(np.uint8)

