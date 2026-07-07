# pyright: reportInvalidTypeForm=false
import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# triton kernel for changing the brightness
@triton.jit
def change_brightness_gpu(pixel_ptr, change_amount: tl.float32, output_ptr, n_pixels, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)

    block_start = pid*BLOCK_SIZE
    pixel_offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = (pixel_offsets < n_pixels)

    # loads the r,g,b values and other makes sure garbage values in mask parts will just be 0
    r = tl.load(pixel_ptr + pixel_offsets * 3 + 0, mask=mask, other=0.0)
    g = tl.load(pixel_ptr + pixel_offsets * 3 + 1, mask=mask, other=0.0)
    b = tl.load(pixel_ptr + pixel_offsets * 3 + 2, mask=mask, other=0.0)

    # outputs for r,g,b clamp function makes sure it is between 0-255
    r_out = tl.clamp(r + change_amount, 0, 255)
    g_out = tl.clamp(g + change_amount, 0, 255)
    b_out = tl.clamp(b + change_amount, 0, 255)

    # store outputs in the output tensor using output_ptr
    tl.store(output_ptr + pixel_offsets * 3 + 0, r_out, mask=mask)
    tl.store(output_ptr + pixel_offsets * 3 + 1, g_out, mask=mask)
    tl.store(output_ptr + pixel_offsets * 3 + 2, b_out, mask=mask)


def change_brightness_gpu_wrapper(image_array, amount):
    # convert the numpy array to a nparray that contains torch floats and selects gpu
    image_tensor = torch.from_numpy(image_array).to(torch.float32).cuda()

    # create empty output tensor in the same shape with the same datatypes as image tensor
    output = torch.empty_like(image_tensor)

    # grab dims of tensor Height, width, color channel
    H, W, C = image_tensor.shape
    # calculate number of pixels
    n_pixels = H*W

    # make the amount to change a 0-dim tensor of float 32
    change_amount = torch.tensor(amount, device="cuda", dtype=torch.float32)

    # how many triton kernels to execute? (ceiling division on num of pixels and block size)
    grid = lambda args: (triton.cdiv(n_pixels, args["BLOCK_SIZE"]),)

    # call to the kernel
    change_brightness_gpu[grid](image_tensor, change_amount, output, n_pixels, BLOCK_SIZE=1024)

    # convert the tensor output into a numpy array so we can convert back to an image
    output_np = output.detach().cpu().numpy().astype(np.uint8)
    return output_np

# triton kernel for changing the contrast
@triton.jit
def change_contrast_gpu(pixel_ptr, change_factor: tl.float32, output_ptr, n_pixels, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)

    block_start = pid * BLOCK_SIZE
    pixel_offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = (pixel_offsets < n_pixels)

    r = tl.load(pixel_ptr + pixel_offsets * 3 + 0, mask=mask, other=0.0)
    g = tl.load(pixel_ptr + pixel_offsets * 3 + 1, mask=mask, other=0.0)
    b = tl.load(pixel_ptr + pixel_offsets * 3 + 2, mask=mask, other=0.0)

    contrast_scale = 1.0 + (change_factor / 100.0)
    r_out = tl.clamp(128.0 + (r - 128.0) * contrast_scale, 0, 255)
    g_out = tl.clamp(128.0 + (g - 128.0) * contrast_scale, 0, 255)
    b_out = tl.clamp(128.0 + (b - 128.0) * contrast_scale, 0, 255)

    tl.store(output_ptr + pixel_offsets * 3 + 0, r_out, mask=mask)
    tl.store(output_ptr + pixel_offsets * 3 + 1, g_out, mask=mask)
    tl.store(output_ptr + pixel_offsets * 3 + 2, b_out, mask=mask)

def change_contrast_gpu_wrapper(image_array, factor):
    image_tensor = torch.from_numpy(image_array).to(torch.float32).cuda()
    output = torch.empty_like(image_tensor)

    H, W, C = image_tensor.shape
    n_pixels = H*W

    change_factor = torch.tensor(factor, device="cuda", dtype=torch.float32)

    grid = lambda args: (triton.cdiv(n_pixels, args["BLOCK_SIZE"]),)

    change_contrast_gpu[grid](image_tensor, change_factor, output, n_pixels, BLOCK_SIZE=1024)

    output_np = output.detach().cpu().numpy().astype(np.uint8)
    return output_np

# triton kernel for changing the gamma contrast
@triton.jit
def change_gamma_contrast_gpu(pixel_ptr, change_factor: tl.float32, output_ptr, n_pixels, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)

    block_start = pid * BLOCK_SIZE
    pixel_offsets = block_start + tl.arange(0,BLOCK_SIZE)

    mask = pixel_offsets < n_pixels

    r = tl.load(pixel_ptr + pixel_offsets * 3 + 0, mask=mask, other=0.0)
    g = tl.load(pixel_ptr + pixel_offsets * 3 + 1, mask=mask, other=0.0)
    b = tl.load(pixel_ptr + pixel_offsets * 3 + 2, mask=mask, other=0.0)
    
    r_normal = r / 255.0
    r_normal_power = libdevice.pow(r_normal, change_factor)
    r_out = tl.clamp(r_normal_power * 255.0, 0, 255)
    g_normal = g / 255.0
    g_normal_power = libdevice.pow(g_normal, change_factor)
    g_out = tl.clamp(g_normal_power * 255.0, 0, 255)
    b_normal = b / 255.0
    b_normal_power = libdevice.pow(b_normal, change_factor)
    b_out = tl.clamp(b_normal_power * 255.0, 0, 255)

    tl.store(output_ptr + pixel_offsets * 3 + 0, r_out, mask=mask)
    tl.store(output_ptr + pixel_offsets * 3 + 1, g_out, mask=mask)
    tl.store(output_ptr + pixel_offsets * 3 + 2, b_out, mask=mask)

def change_gamma_contrast_gpu_wrapper(image_array, factor):
    image_tensor = torch.from_numpy(image_array).to(torch.float32).cuda()
    output = torch.empty_like(image_tensor)

    H, W, C = image_tensor.shape
    n_pixels = H*W

    change_factor = torch.tensor(factor, device="cuda", dtype=torch.float32)

    grid = lambda args: (triton.cdiv(n_pixels, args["BLOCK_SIZE"]),)

    change_gamma_contrast_gpu[grid](image_tensor, change_factor, output, n_pixels, BLOCK_SIZE=1024)

    output_np = output.detach().cpu().numpy().astype(np.uint8)

    return output_np

