# Copyright 2022 Huawei Technologies Co., Ltd
# CREATED:  2020-6-04 20:12:13
# MODIFIED: 2022-08-17 13:48:45
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
This script provide postprocessing for inference
Copyright 2022 Huawei Technologies Co., Ltd
"""
# -*- coding:utf-8 -*-
# import acl
import time
import numpy as np

from data.constant import ACL_MEMCPY_DEVICE_TO_HOST


# def get_model_output_by_index(model_output, i):
#     temp_output_buf = acl.mdl.get_dataset_buffer(model_output, i)

#     infer_output_ptr = acl.get_data_buffer_addr(temp_output_buf)
#     infer_output_size = acl.get_data_buffer_size(temp_output_buf)

#     output_host, _ = acl.rt.malloc_host(infer_output_size)
#     acl.rt.memcpy(output_host, infer_output_size, infer_output_ptr,
#                           infer_output_size, ACL_MEMCPY_DEVICE_TO_HOST)

#     return acl.util.ptr_to_numpy(output_host, (infer_output_size//4,), 11)


def detect(features, model_shape, class_num):
    """
    x(bs,3,20,20,85)
    """
    z = []
    for i in range(3):
        y = features[i]
        y[..., 0:2] = y[..., 0:2] * model_shape[1] # xy
        y[..., 2:4] = y[..., 2:4] * model_shape[0] # wh
        z.append(y.reshape(1, -1, class_num))
    return np.concatenate(z, 1)


def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.6, agnostic=False):
    """Performs Non-Maximum Suppression (__nms) on inference results

    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    time_limit = 10.0  # seconds to quit after

    t = time.time()
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = __xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        i, j = np.nonzero(x[:, 5:] > conf_thres)
        x = np.concatenate((box[i], x[i, j + 5, None], j[:, None].astype('float32')), 1)

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue
            
        # Sort by confidence
        # x = x[x[:, 4].argsort(descending=True)]

        # Batched __nms
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = __nms(boxes, scores, iou_thres)
       
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
            
        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    __clip_coords(coords, img0_shape)
    return coords


def __xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    #y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    
    return y


def __nms(boxes, scores, iou_thres):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return np.zeros(1)
    
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float32")
    
    # initialize the list of picked indexes	
    pick = []
    
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)
    
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > iou_thres)[0])))
        
    # return only picked value
    return np.array(pick)


def __clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    np.clip(boxes[:, 0], 0, img_shape[1])
    np.clip(boxes[:, 1], 0, img_shape[0])
    np.clip(boxes[:, 2], 0, img_shape[1])
    np.clip(boxes[:, 3], 0, img_shape[0])