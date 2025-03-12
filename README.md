# Adaconv-ONNX
使得adaconv可以转换为onnx
## <p>运行指令大全：</p>
- python train.py -c ./configs/lambda100.yaml -d ./data/raw -l ./logs<br>
- tensorboard --logdir=./logs/tensorboard</br>
- python test.py --config ./configs/lambda100.yaml --model_ckpt ./logs/ckpts/last.pt --content_path ./Test/input --style_path ./Test/style --output_path output.jpg</br>
- python onnx_exporter.py --checkpoint ./logs/ckpts/last.pt --config ./configs/lambda100.yaml --output model.onnx</br>
- python onnx_validator.py --onnx model.onnx --input-shape "input_name:8,3,256,256" "content:8,3,256,256" "style:8,3,256,256"</br>