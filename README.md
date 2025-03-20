# Adaconv-ONNX
使得adaconv可以转换为onnx
## <p>运行指令大全：</p>
- python train.py -c ./configs/lambda100.yaml -d ./data/raw -l ./logs<br>
- tensorboard --logdir=./logs/tensorboard</br>

- python visualization_script.py</br>
- python export_script.py --output model.onnx  </br>

- python onnx_validator.py --onnx model.onnx --input-shape "input_name:8,3,256,256" "content:8,3,256,256" "style:8,3,256,256"</br>
- python onnx_validator.py --onnx model.onnx --input-shape "input_name:1,3,256,256" "content:1,3,256,256" "style:1,3,256,256"</br>