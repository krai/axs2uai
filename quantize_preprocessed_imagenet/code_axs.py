import os
import bisect
import numpy as np

def dequantize_value(val, bias):
    mntmask = 15
    expmask = 112
    sgnmask = 128

    mnt = val & mntmask
    exp = (val & expmask) >> 4
    sgn = -1.0 if (val & sgnmask) else 1.0

    mnt = (16 | mnt) if exp else mnt
    exp = exp if exp else 1

    return np.float32(sgn * mnt * pow(2, int(exp) + bias))

def process_image(filename, preprocessed_dataset_path, full_output_path, resolution, given_channel_means, sorted_quant_bins, sorted_quant_bounds):
    file_path = os.path.join(preprocessed_dataset_path, filename)
    output_file_path = os.path.join(full_output_path, filename)

    img = np.fromfile(file_path, dtype=np.uint8)
    img = np.reshape(img, (resolution, resolution, 3))
    img = img.astype(np.float32)
    means = np.array(given_channel_means, dtype=np.float32)
    img -= means

    quantized_img = np.array([sorted_quant_bounds[bisect.bisect_right(sorted_quant_bins, val), 1] for val in img.flatten()], dtype=np.uint8)
    quantized_img = np.reshape(quantized_img, img.shape)
    quantized_img = np.where(img != 0, quantized_img, np.uint8(0))

    quantized_img.tofile(output_file_path)

def quantize(preprocessed_dataset_path, newborn_entry_path, quantized_dtype, rel_file_path, 
             bias, given_channel_means, resolution):
    if quantized_dtype != "FP8p":
        raise ValueError('Unsupported quantized data type is used.')

    bias = int(bias)

    quant_bounds = []
    for quantVal in range(256):
        floatVal = dequantize_value(np.uint8(quantVal), bias)
        quant_bounds.append((floatVal, np.uint8(quantVal)))

    sorted_quant_bounds = np.array(sorted(quant_bounds, key=lambda x: x[0]))

    for i in range(len(sorted_quant_bounds)-1):
        sorted_quant_bounds[i][0] = (sorted_quant_bounds[i][0] + sorted_quant_bounds[i+1][0]) / 2
    sorted_quant_bounds[-1][0] = np.finfo(np.float32).max

    sorted_quant_bins = list(sorted_quant_bounds[:, 0])

    full_output_path = os.path.join(newborn_entry_path, rel_file_path)
    if not os.path.exists(full_output_path):
        os.makedirs(full_output_path)
    
    image_list_file = os.path.join(preprocessed_dataset_path, 'image_list.txt')
    image_list_output_file = os.path.join(full_output_path, 'image_list.txt')
    if os.path.exists(image_list_file):
        with open(image_list_file, 'rb') as source_file:
            with open(image_list_output_file, 'wb') as destination_file:
                destination_file.write(source_file.read())

    image_files = [f for f in os.listdir(preprocessed_dataset_path) if f != 'image_list.txt']

    for filename in image_files:
        process_image(filename, preprocessed_dataset_path, full_output_path, resolution, given_channel_means, sorted_quant_bins, sorted_quant_bounds)

    return
