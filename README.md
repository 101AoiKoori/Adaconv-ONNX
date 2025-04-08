# Adaconv-ONNX
使得adaconv可以转换为onnx
## <p>运行指令大全：</p>
### <p>运行</p>
#### 训练
- python train.py -c ./configs/lambda100.yaml -d ./data/raw -l ./logs</br>
#### 微调
- python train.py -c ./configs/lambda100.yaml -d ./data/256 --finetune</br>
---
### <p>查看数据</p>
- tensorboard --logdir=./logs/tensorboard</br>
- tensorboard --logdir=./logs/finetune/tensorboard</br>
- python script_visualization.py</br>
---
### <p>导出ONNX</p>
- python script_export.py --output model.onnx</br>
---
### <p>测试ONNX</p>
- python script_visualization.py  or  python onnx_validator.py --onnx model.onnx --input-shape "content:1,3,256,256" "style:1,3,256,256"</br>
---
### <p>测试模型</p>
- python test.py</br>
- Python cuda_test.py</br>
