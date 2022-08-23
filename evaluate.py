# Copyright 2021 Huawei Technologies Co., Ltd
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
"""YoloV5 eval."""
import os
import time
import argparse
import numpy as np
from glob import glob
import pandas as pd

import mindspore as ms

from src.yolo import YOLOV5
from src.util import DetectionEngine
from src.yolo_dataset import create_yolo_dataset
import datetime
from model_utils.config import config

# only useful for huawei cloud modelarts
from model_utils.moxing_adapter import moxing_wrapper, modelarts_pre_process


# Log Functions / CYAN
def LOG(inp):
    config.logger.info(f'[\033[96mLOG\033[0m] {inp}')

# Success Functions / GREEN
def SUCCESS(inp):
    config.logger.info(f'\033[92mSUCCESS\033[0m {inp}')

# Process Functions / YELLOW
def PROCESS(inp):
    config.logger.info(f'\033[33mPROCESS\033[0m {inp}')

# Loading checkpoint and placing into model. (Model = YOLOV5, File = .ckpt file which selected)
def load_parameters(network, filename):
    LOG("yolov5 pretrained network model: %s"% (filename))
    param_dict = ms.load_checkpoint(filename)
    param_dict_new = {}
    for key, values in param_dict.items():
        if key.startswith('moments.'):
            continue
        elif key.startswith('yolo_network.'):
            param_dict_new[key[13:]] = values
        else:
            param_dict_new[key] = values
    ms.load_param_into_net(network, param_dict_new)
    LOG('load_model %s success'% (filename))


@moxing_wrapper(pre_process=modelarts_pre_process, pre_args=[config])
def run_eval(data_root = '/tmp/workspace/COCO2017/train/val2017',
            ann_file ="/tmp/workspace/COCO2017/train/annotations/instances_val2017.json",
            yolov5_version = "yolov5s",
            device = 'CPU',
            ckpt_file = '/tmp/mindspore/model',
            batch_limitter = 10,
            epoches = 1,
            per_batch_size = 32,
            test_img_shape = [640, 640],
            test_ignore_threshold =  0.001,
            config_path = None
            ):
    
    # Path to files from config file data_root file and annotations file
    if config_path:
        LOG(f'=================CONFIG MODE ON=================')
        
        data_root = config.eval_data_dir
        LOG(f'Data Obtained from {data_root}')
        
        
        ann_file = config.ann_file
        LOG(f'Annotations File Obtained from {ann_file}')
        
        # Path to your pretrained confil folder
        ckpt_file = config.pretrained
        LOG(f'Checkpoints Folder Obtained from {ckpt_file}')
        
        # Your device from config file
        # device = config.device_target
        
        # version of yolov5 like yolov5s
        yolov5_version = config.yolov5_version
        
        # length of per batch size
        per_batch_size = config.per_batch_size
        
        # shape of test image
        test_img_shape =config.test_img_shape
        
        # treshold to be ignored
        test_ignore_threshold = config.test_ignore_threshold
        
    

    # Selecting Huawei Ascend Device to run all evaluation process
    LOG(f'Device is \033[33m{str(device)}\033[0m')

    ms.set_context(mode = ms.GRAPH_MODE, device_target = device)
        
    SUCCESS(f'Parmer Setup Sucess')
    start_time = time.time()
    
    # Network Creation
    LOG('Netwotk is Creating for Current .ckpt Evaluetion')
    dict_version = {'yolov5s': 0, 'yolov5m': 1, 'yolov5l': 2, 'yolov5x': 3}
    # Calling YOLOv5 Model to update weights with selected ckpt file
    network = YOLOV5(is_training = False, version = dict_version[yolov5_version])
    
    # Taking ckpt file by looking its extension, otherwise it takes latest one in the folder
    if ckpt_file[-4:] == 'ckpt':
        LOG(f'Your .ckpt File is {ckpt_file}')
        pass
    else:
        print(f'\n\n\n===\n{ckpt_file}\n===\n\n\n')
        ckpt_file = sorted(glob(f'{ckpt_file}/*.ckpt'), key=os.path.getmtime)[-1]
        LOG(f'Your .ckpt Folder is {ckpt_file}')

    if os.path.isfile(ckpt_file):
        load_parameters(network, ckpt_file)
    else:
        raise FileNotFoundError(f"{ckpt_file} is not a filename.")
        
    LOG('Dataset Creating')

    ds = create_yolo_dataset(data_root, ann_file, is_training=False, batch_size=per_batch_size,
                             device_num=1, rank=0, shuffle=False, config=config)
    
    
    LOG(f'Shape of Test File is: {test_img_shape}')
    LOG('Total %d Images to Eval'% (ds.get_dataset_size() * per_batch_size))


    # Changing Model Mode Train to False for Inference
    network.set_train(False)
    

    # Calling detection engine to test all process
    detection = DetectionEngine(config, config.test_ignore_threshold)
    

    # Setting up the input shape of the model
    input_shape = ms.Tensor(tuple(test_img_shape), ms.float32)


    # INFERENCE EXECUTION PART
    LOG(f'Inference Begins...')
    batches_track = 0
    
    for index, data in enumerate(ds.create_dict_iterator(output_numpy=True, num_epochs=1)):

        image = data["image"]
        # Shaping data to corresponding input format
        image = np.concatenate((image[..., ::2, ::2], image[..., 1::2, ::2],
                                image[..., ::2, 1::2], image[..., 1::2, 1::2]), axis=1)

        # Changing image array into Tensor(Like pytorch Tensor and numpys np.array) and process all
        image = ms.Tensor(image)
        image_shape_ = data["image_shape"]
        image_id_ = data["img_id"]
        output_big, output_me, output_small = network(image, input_shape)
        output_big = output_big.asnumpy()
        output_me = output_me.asnumpy()
        output_small = output_small.asnumpy()

        # Detection part
        detection.detect([output_small, output_me, output_big], per_batch_size, image_shape_, image_id_)
        batches_track += 1

        # Limiting batches to create test result with limited image to process faster
        if batches_track == batch_limitter:
            break

        # Printing process every 10 step with adhjusted percentage
        if index % 2 == 0:
            PROCESS(f'Current Process: %{index / batch_limitter * 100:.2f} done')
    PROCESS(f'Current Process: %100 done!!!')
    print('='*50)

    # Mean Absolute Precision Calculation with outputs. This process took longer than others
    LOG(f'mAP is Calculating... Note: This process may take a while.')
    detection.do_nms_for_results()
    result_file_path = detection.write_result()

    # Getting evaluated result 
    LOG('File Path of the Reult: %s'% (result_file_path))
    eval_result = detection.get_eval_result()

    # Write output to txt file
    with  open("output.txt", "w") as file:
        file.write(eval_result)
        file.close()
        
    # Save output as JSON
    if os.path.exists("./output/evals.json"):
        df = pd.read_json('./output/evals.json')
        new = pd.read_csv('output.txt', names = [f'epoches_{epoches}'], sep=' = ',  header=None, index_col=0)[f'epoches_{epoches}'].values
        df[f'epoches_{epoches}'] = new
        df = df.T
        df.to_json(r'./output/evals.json', orient='index')
    else:
        df = pd.read_csv('./output.txt', names = [f'epoches_{epoches}'], sep=' = ',  header=None, index_col=0).T
        df.to_json(r'./output/evals.json', orient='index')
    
    # Remove thrash txt file
    if os.path.exists("./output.txt"):
          os.remove("./output.txt")

    # Displaying output of the result on terminal
    cost_time = time.time() - start_time
    eval_log_string = '\n================== Eval Result of the Process ==================\n' + eval_result
    LOG(eval_log_string)
    LOG('testing cost time %.2f h'% (cost_time / 3600.))

# Argument parsing 
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default = None, help = 'Use config file for variables')
    parser.add_argument('--ckpt_file', default = '/tmp/mindspore/model', help = 'PAth to ckpt folder Note: Model selects latest code by automaticly. If you give ckpt file, it will take that')
    opt = parser.parse_args()
    print(opt)
    return opt

def main(opt):
    run_eval(**vars(opt))
    
if __name__ == "__main__":    
    opt = parse_opt()
    main(opt)