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
"""calculation of total evaluation time"""

start_time="$(date -u +%s)"
python3 ../evaluate.py --config_path ../default_config.yaml
end_time="$(date -u +%s)"
elapsed="$(($end_time-$start_time))"
echo "Total of $elapsed seconds elapsed for process"