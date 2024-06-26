# Copyright 2021 Huawei Technologies Co., Ltd
# CREATED:  2022-11-25 10:12:13
# MODIFIED: 2022-12-05 12:48:45
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
# =========================================================================
"""YoloV5 train"""
import os
import time
import mindspore as ms
import mindspore.nn as nn
import mindspore.communication as comm

from src.logger import get_logger
from src.lr_scheduler import get_lr
from src.util import DetectionEngine
from src.yolo import YOLOV5, YoloWithLossCell
from src.yolo_dataset import create_yolo_dataset
from src.util import AverageMeter, get_param_groups, cpu_affinity
from src.initializer import default_recurisive_init, load_yolov5_params

from model_utils.config import config
from model_utils.device_adapter import get_device_id

# only useful for huawei cloud modelarts.
from evaluate import run_eval
from model_utils.moxing_adapter import moxing_wrapper, modelarts_pre_process, modelarts_post_process


# yolo version dictionary
dict_version = {'yolov5s': 0, 'yolov5m': 1, 'yolov5l': 2, 'yolov5x': 3}

ms.set_seed(1)

def init_distribute():
    comm.init()
    config.rank = comm.get_rank()
    config.group_size = comm.get_group_size()
    ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                 device_num=config.group_size)


def train_preprocess():
    if config.lr_scheduler == 'cosine_annealing' and config.max_epoch > config.T_max:
        config.T_max = config.max_epoc
    
    # training configs
    config.lr_epochs = list(map(int, config.lr_epochs.split(',')))
    config.train_img_dir = os.path.join(config.data_dir, config.train_img_dir)
    config.train_json_file = os.path.join(config.data_dir, config.train_json_file)
    device_id = get_device_id()
    ms.set_context(mode=ms.GRAPH_MODE, device_target=config.device_target, device_id=device_id)

    # evaluation configs
    config.eval_img_dir = os.path.join(config.data_dir, config.eval_img_dir)
    config.eval_json_file = os.path.join(config.data_dir, config.eval_json_file)

    if config.is_distributed:
        # init distributed
        init_distribute()

    # for promoting performance in GPU device
    if config.device_target == "GPU" and config.bind_cpu:
        cpu_affinity(config.rank, min(config.group_size, config.device_num))

    # logger module is managed by config, it is used in other function. e.x. config.logger.info("xxx")
    config.logger = get_logger(config.output_dir, config.rank)
    config.logger.save_args(config)


def train_eval_setup(data_root, ann_file):
    network = YOLOV5(is_training = False, version = dict_version[config.yolov5_version])

    ds = create_yolo_dataset(image_dir=data_root, anno_path=ann_file, is_training=False, 
                            batch_size=config.train_per_batch_size, device_num=1, rank=0, shuffle=False, config=config)

    # Calling detection engine to test all process
    detection = DetectionEngine(config, config.test_ignore_threshold)

    # Setting up the input shape of the model
    input_shape = ms.Tensor(tuple(config.test_img_shape), ms.float32)
    
    return network, detection, ds, input_shape


@moxing_wrapper(pre_process=modelarts_pre_process, post_process=modelarts_post_process, pre_args=[config])
def run_train():
    train_preprocess()

    loss_meter = AverageMeter('loss')
    
    network = YOLOV5(is_training=True, version=dict_version[config.yolov5_version])
    # default is kaiming-normal
    default_recurisive_init(network)
    load_yolov5_params(config, network)
    network = YoloWithLossCell(network)

    ds = create_yolo_dataset(image_dir=config.train_img_dir, anno_path=config.train_json_file, is_training=True,
                             batch_size=config.train_per_batch_size, device_num=config.group_size, rank=config.rank, config=config)
    config.logger.info('Finish loading dataset')

    # Evaluation 
    # Calling YOLOv5 Model to update weights with selected ckpt file
    if config.eval_per_step > 0:
        network_eval_params = train_eval_setup(config.eval_img_dir, config.eval_json_file)

    steps_per_epoch = ds.get_dataset_size()
    lr = get_lr(config, steps_per_epoch)
    opt = nn.Momentum(params=get_param_groups(network), momentum=config.momentum, learning_rate=ms.Tensor(lr),
                      weight_decay=config.weight_decay, loss_scale=config.loss_scale)
    network = nn.TrainOneStepCell(network, opt, config.loss_scale // 2)
    network.set_train()

    data_loader = ds.create_tuple_iterator(do_copy=False)
    first_step = True
    t_end = time.time()

    #region MindInsight
    # If you are using multiple NPU training scenario you should add this 2 line of code. 
    # It is provides to see your all summaries in one directory. You can switch directories easily while using mindInsight visualizer.
    # from mindspore.communication import get_rank
    # summary_dir = "../summary/summary_dir" + str(get_rank()) # It will be created in the scripts/ directory.
    summary_dir = '../summary/summary_dir/' # single NPU
    #endregion
    with ms.train.summary.SummaryRecord(summary_dir, network=network) as summary_record:
        for epoch_idx in range(config.max_epoch):
            for step_idx, data in enumerate(data_loader):
                images = data[0]
                input_shape = images.shape[2:4]
                input_shape = ms.Tensor(tuple(input_shape[::-1]), ms.float32)
                loss = network(images, data[2], data[3], data[4], data[5], data[6],
                               data[7], input_shape)
                loss_meter.update(loss.asnumpy())
                                           
                # it is used for loss, performance output per config.log_interval steps.
                if (epoch_idx * steps_per_epoch + step_idx) % config.log_interval == 0:
                    time_used = time.time() - t_end
                    # parameters to be displayed on the WebUI
                    summary_record.set_mode('train')
                    summary_record.add_value('scalar', 'loss', loss)
                    summary_record.add_value('scalar', 'dummyLoss', loss+40)
                    # You can use add_value for adding new variables to watch. validationLoss, ... etc.
                    summary_record.record(step_idx + 1)

                    if first_step:
                        fps = config.train_per_batch_size * config.group_size / time_used
                        per_step_time = time_used * 1000
                        first_step = False
                    else:
                        fps = config.train_per_batch_size * config.log_interval * config.group_size / time_used
                        per_step_time = time_used / config.log_interval * 1000
                    config.logger.info('epoch[{}], iter[{}], {}, fps:{:.2f} imgs/sec, '
                                       'lr:{}, per step time: {}ms'.format(epoch_idx + 1, step_idx + 1,
                                                                         loss_meter, fps, lr[step_idx], per_step_time))
                    t_end = time.time()
                    loss_meter.reset()
                    
            if config.rank == 0:
                ckpt_name = os.path.join(config.output_dir, "yolov5_{}_{}.ckpt".format(epoch_idx + 1, steps_per_epoch))
                ms.save_checkpoint(network, ckpt_name)
                
            if network_eval_params:
                # Precision and Recall Scores Which Returned From Evaluation
                if epoch_idx%config.eval_per_step == 0:   
                    try:
                        eval_scores = run_eval(epoches = epoch_idx,
                                            network_params = network_eval_params)
                    except:
                        config.logger.info('error occured during evaluation process...')
    config.logger.info('==========end training===============')


if __name__ == "__main__":
    run_train()