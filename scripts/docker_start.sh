#!/bin/bash
# Copyright 2022 Huawei Technologies Co., Ltd
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
# ============================================================================
"""easy start bash script for docker run"""

docker_image=$1
data_dir=$2
model_dir=$3

docker run -it --ipc=host \
              --device=/dev/davinci0 \
              --device=/dev/davinci1 \
              --device=/dev/davinci2 \
              --device=/dev/davinci3 \
              --device=/dev/davinci4 \
              --device=/dev/davinci5 \
              --device=/dev/davinci6 \
              --device=/dev/davinci7 \
              --device=/dev/davinci_manager \
              --device=/dev/devmm_svm --device=/dev/hisi_hdc \
              -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
              -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
              -v ${model_dir}:${model_dir} \
              -v ${data_dir}:${data_dir}  \
              -v /root/ascend/log:/root/ascend/log ${docker_image} /bin/bash