# Copyright 2024 Untether AI Inc. ALL RIGHTS RESERVED.
#
# This source code (Information) is proprietary to Untether AI Inc.
# (Untether) and MAY NOT be copied by any method, incorporated into or
# linked to another program, reverse engineered or otherwise used in
# any manner whatsoever without the express prior written consent of
# Untether. This Information and any portion thereof is and shall
# remain the property of Untether. Untether makes no representation
# or warranty regarding the accuracy of the Information nor its fitness nor
# merchantability for any purpose.  Untether assumes no responsibility
# or liability for its use in any way and conveys to the user no license,
# right or title under any patent, copyright or otherwise, and makes
# no representation or warranty that this Information does not infringe
# any third party patent, copyright or other proprietary right.

from pathlib import Path
import warnings
import sys
import json
import logging
import numpy as np
from tqdm import tqdm
import torch
from torchvision import models
import onnx
import cv2

from untetherai.model_creator.kernel_framework.core import Network
from untetherai.model_executor.end_to_end import compile_network
from untetherai.model_garden.resnet import resnet50
from untetherai.model_quantizer import dtypes, from_torch, ops, q_ops, quantize, fx_to_onnx, onnx_to_fx
from untetherai.model_quantizer.acquisition import ChannelsLastModule
from untetherai.model_quantizer.importer import import_onnx

logging.getLogger().setLevel(logging.ERROR)
warnings.simplefilter("ignore", UserWarning)

args = iter(sys.argv[1:])
refresh_rate = int(next(args))
seed = int(next(args))
newborn_entry_path = Path(next(args))
original_model_path = Path(next(args))
dataset_path = Path(next(args))
calib_list_path = Path(next(args))

quantized_model_path = newborn_entry_path / "quantized_model"

def center_crop(img, out_height, out_width):
    height, width, _ = img.shape
    left = int((width - out_width) / 2)
    right = int((width + out_width) / 2)
    top = int((height - out_height) / 2)
    bottom = int((height + out_height) / 2)
    img = img[top:bottom, left:right]
    return img


def resize_with_aspectratio(img, out_height, out_width, scale=87.5, inter_pol=cv2.INTER_LINEAR):
    height, width, _ = img.shape
    new_height = int(100.0 * out_height / scale)
    new_width = int(100.0 * out_width / scale)
    if height > width:
        w = new_width
        h = int(new_height * height / width)
    else:
        h = new_height
        w = int(new_width * width / height)
    img = cv2.resize(img, (w, h), interpolation=inter_pol)
    return img

def pre_process_vgg(img, dims=(224, 224, 3), need_transpose=False):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    output_height, output_width, _ = dims
    cv2_interpol = cv2.INTER_AREA
    img = resize_with_aspectratio(img, output_height, output_width, inter_pol=cv2_interpol)
    img = center_crop(img, output_height, output_width)
    img = np.asarray(img, dtype="float32")

    # normalize image
    means = np.array([123.68, 116.78, 103.94], dtype=np.float32)
    img -= means

    # transpose if needed
    if need_transpose:
        img = img.transpose([2, 0, 1])
    return img

def get_calibration_data(root, files, batch_size):
    class CalibrationDataset(torch.utils.data.Dataset):
        def __init__(self, root, files):
            with open(files, "r") as f:
                self.files = []
                for line in f.read().splitlines():
                    file, *_ = line.split()
                    (choice,) = root.glob(f"*/{file}")
                    self.files.append(str(choice))

        def __getitem__(self, idx):
            return pre_process_vgg(cv2.imread(self.files[idx]))

        def __len__(self):
            return len(self.files)

    dataset = CalibrationDataset(root, files)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, pin_memory=True)

@torch.inference_mode()
def generate(refresh_rate, seed, original_model_path, dataset_path, calib_list_path, quantized_model_path):

    quantized_model_path.mkdir()

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    onnx_model = onnx.load(original_model_path)
    subgraph = [["input_tensor:0"], ["resnet_model/dense/BiasAdd:0"]]
    onnx_model = onnx.utils.Extractor(onnx_model).extract_model(*subgraph)
    float_model = onnx_to_fx(onnx_model).eval()
    float_model.initializers.onnx_initializer_0 = float_model.initializers.onnx_initializer_0[:, 1:]
    float_model.initializers.onnx_initializer_1 = float_model.initializers.onnx_initializer_1[1:]
    float_model = ChannelsLastModule(float_model, (2,), 0)

    calib_data = get_calibration_data(dataset_path, calib_list_path, 10)

    example_x = next(iter(calib_data)).to(device)
    config = q_ops.default_config(output_dtype=dtypes.FP8p).register(
        ops.convolution,
        q_ops.convolution(dtypes.FP8p, dtypes.FP8p, per_tensor=True, restrict_shift=True),
    )
    observing = from_torch(float_model, (example_x,), config)
    for images in tqdm(calib_data):
        observing.run(images.to(device))
    quant_model = quantize(observing).to(device)

    qdtype = quant_model.quantize_args_0.y_params
    bias = list(qdtype.dtype.offsets)[0] - qdtype.bias.item()
    with open(newborn_entry_path / "data_axs.json") as file:
        entry = json.load(file)
    entry["bias"] = bias
    with open(newborn_entry_path / "data_axs.json", "w") as file:
        json.dump(entry, file)

    _, onnx_model = fx_to_onnx(quant_model, propagate_meta_data=False)

    reloaded_graph = import_onnx(onnx_model)
    net_params = resnet50.extract_resnet_parameters_from_ir(reloaded_graph)

    network = Network()
    resnet50.instantiate(network, net_params, refresh_rate)

    resnet50.generate_constraints(network)

    compile_network(network, quantized_model_path)

if __name__ == "__main__":
    generate(refresh_rate, seed, original_model_path, dataset_path, calib_list_path, quantized_model_path)
