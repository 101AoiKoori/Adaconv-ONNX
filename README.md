# Adaconv-ONNX
使得adaconv可以转换为onnx
## <p>运行指令大全：</p>
- python train.py -c ./configs/lambda100.yaml -d ./data/raw -l ./logs<br>

- tensorboard --logdir=./logs/tensorboard</br>

- python script_visualization.py  or  python onnx_validator.py --onnx model.onnx --input-shape "content:1,3,256,256" "style:1,3,256,256"</br>

- python script_export.py --output SourceAdaConv.onnx</br>

- python test.py</br>

- Python cuda_test.py</br>
