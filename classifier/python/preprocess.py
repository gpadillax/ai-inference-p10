# Copyright 2022 IBM All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Method to preprocess the image so that it can be fed to the resnet model.

   This code is a modified version of the examples provided here:
   https://github.com/onnx/models/tree/main/vision/classification/resnet#preprocessing
"""
import numpy as np
from PIL import Image


def preprocess(input_image: Image.Image) -> np.ndarray:
    """Preprocesses an image so that it can be provided as input to RESNET"""
    image = input_image.resize((224, 224))
    img_data = np.array(image)
    img_data = np.moveaxis(img_data, -1, 0)  # channels first

    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])

    norm_img_data = np.zeros(img_data.shape).astype("float32")
    for i in range(img_data.shape[0]):
        # for each pixel in each channel, divide the value by 255
        # to get value between [0, 1] and then normalize
        norm_img_data[i, :, :] = (img_data[i, :, :] / 255 - mean_vec[i]) / stddev_vec[i]
    return norm_img_data
