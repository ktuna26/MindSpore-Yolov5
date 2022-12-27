# PyACL YoloV5 Asynchronous Object Detection

Please open the jupyter notebook for a quick demo. This sample uses  **MindSPore Yolov5**.  

## Original Network Link

https://gitee.com/mindspore/models/tree/master/official/cv/yolov5

## CKPT model -> AIR format -> Ascend om format
### CKPTH -> AIR

Use the .../export.py script in this repository to convert CKPT file to AIR file.  

```
python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format AIR
```

### AIR -> OM
```bash
atc --framework=1 \
    --model="weights/yolov5.air" \
    --output="weights/yolov5" \
    --soc_version=Ascend310
```

## Resources
[1] https://www.hiascend.com/document/detail/en/canncommercial/504/inferapplicationdev/aclpythondevg/aclpythondevg_02_0023.html

## Copyright
Huawei Technologies Co., Ltd

## License
MIT