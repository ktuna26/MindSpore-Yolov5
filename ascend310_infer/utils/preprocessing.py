# Copyright 2022 Huawei Technologies Co., Ltd
# CREATED:  2020-6-04 20:12:13
# MODIFIED: 2022-08-17 12:05:45
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
This script provide preprocessing for inference
Copyright 2021 Huawei Technologies Co., Ltd
"""
# -*- coding:utf-8 -*-
import numpy as np
import cv2


def __letterbox(img, new_shape=(640, 640), color=(128, 128, 128), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def __focus_process(x):
    # x(b,c,w,h) -> y(b,4c,w/2,h/2)
    return np.concatenate([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)


def preprocess(img_path, model_input_size):
    img = cv2.imread(img_path) # org_bgr image
    img_dims = img.shape
    img_resized = __letterbox(img[:, :, ::-1], model_input_size)[0] # bgr to rgb (color space change) & resize
    img_resized = img_resized.transpose(2, 0, 1) # [h, w, c] to [c, h, w]
    print("[PreProc] img_resize shape:", img_resized.shape)

    image_np_expanded = np.expand_dims(img_resized, axis=0)  # NCHW
    image_np_expanded = image_np_expanded.astype('float32') / 255.0 # Converts the image pixels to the range [-1, 1]
    print("[PreProc] image_np_expanded shape:", image_np_expanded.shape)
    
    # Focus
    img_numpy = np.ascontiguousarray(__focus_process(image_np_expanded))
    print("[PreProc] img_numpy shape:", img_numpy.shape)

    return img_numpy, img_dims