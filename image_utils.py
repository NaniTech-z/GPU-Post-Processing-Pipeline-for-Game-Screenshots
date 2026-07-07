from PIL import Image # importing the pillow library
import numpy as np 

# function to load an image and converts into a numpy array
# format [height, width, color channel]
def load_image_to_array(image):
    im = Image.open(image).convert("RGB")
    im_array = np.array(im)
    return im_array

# function to save an image
def save_image(path, im_array):
    im = Image.fromarray(im_array)
    im.save(path)