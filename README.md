# Adaconv-ONNX
使得adaconv可以转换为onnx
## <p>运行指令大全：</p>
### <p>运行</p>
#### 训练
- python train.py -c ./configs/lambda100.yaml -d ./data/raw -l ./logs</br>
#### 微调
- python train.py -c ./configs/lambda100.yaml -d ./data/finetune --finetune</br>
---
### <p>查看数据</p>
- tensorboard --logdir=./logs/tensorboard</br>
- tensorboard --logdir=./logs/finetune/tensorboard</br>
- python visualizer.py</br>
---
### <p>导出ONNX</p>
- python exporter.py --output model.onnx  # Use default settings</br>
- python exporter.py --output model.onnx --static  # Static mode</br>
- python exporter.py --output model.onnx --dynamic  # Fully dynamic mode</br>
- python exporter.py --output model.onnx --dynamic-batch  # Dynamic batch size</br>
