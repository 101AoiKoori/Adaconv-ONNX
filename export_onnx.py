
import torch
import argparse
import warnings
import onnx
from onnx import shape_inference
from onnxruntime import InferenceSession, SessionOptions

warnings.filterwarnings("ignore", category=FutureWarning, module="torch.onnx.utils")

def validate_cuda_availability():
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，请检查您的环境或使用 CPU 模式")
        return False
    print(f"✅ CUDA可用，当前设备: {torch.cuda.get_device_name(0)}")
    return True

def check_reshape_nodes(onnx_model):
    all_reshape_valid = True
    for node in onnx_model.graph.node:
        if node.op_type == 'Reshape':
            print(f"正在检查 Reshape 节点: {node.name}")
            # 获取输入张量信息
            input_tensor = next((i for i in onnx_model.graph.input if i.name == node.input[0]), None)
            if input_tensor is None:
                # 尝试从中间张量中查找
                input_tensor = next((t for t in onnx_model.graph.value_info if t.name == node.input[0]), None)
            # 获取输出张量信息
            output_tensor = next((o for o in onnx_model.graph.output if o.name == node.output[0]), None)
            if output_tensor is None:
                # 尝试从中间张量中查找
                output_tensor = next((t for t in onnx_model.graph.value_info if t.name == node.output[0]), None)

            if input_tensor and output_tensor:
                input_shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
                output_shape = [dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]
                print(f"输入形状: {input_shape}")
                print(f"输出形状: {output_shape}")

                # 计算输入和输出的有效元素数量
                input_size = 1
                unknown_input_dims = []
                for i, dim in enumerate(input_shape):
                    if dim > 0:
                        input_size *= dim
                    elif dim == 0:
                        print(f"  输入形状的第 {i} 维是动态的（值为 0），假设该维可以根据其他维度调整。")
                    else:
                        unknown_input_dims.append(i)
                        print(f"  输入形状的第 {i} 维未知（值为负数），可能导致形状计算不准确。")

                output_size = 1
                unknown_output_dims = []
                for i, dim in enumerate(output_shape):
                    if dim > 0:
                        output_size *= dim
                    elif dim == 0:
                        print(f"  输出形状的第 {i} 维是动态的（值为 0），假设该维可以根据其他维度调整。")
                    else:
                        unknown_output_dims.append(i)
                        print(f"  输出形状的第 {i} 维未知（值为负数），可能导致形状计算不准确。")

                if len(unknown_input_dims) == 0 and len(unknown_output_dims) == 0:
                    if input_size != output_size:
                        print(f"❌ Reshape 节点 {node.name} 存在问题: 输入形状 {input_shape} 无法重塑为 {output_shape}，输入元素数量 {input_size} 不等于输出元素数量 {output_size}。")
                        all_reshape_valid = False
                    else:
                        print(f"✅ Reshape 节点 {node.name} 正常: 输入元素数量 {input_size} 等于输出元素数量 {output_size}。")
                else:
                    print(f"⚠️ Reshape 节点 {node.name} 存在未知维度，无法准确判断是否可以重塑。输入未知维度: {unknown_input_dims}，输出未知维度: {unknown_output_dims}。")
            else:
                print(f"❌ 无法找到 Reshape 节点 {node.name} 的输入或输出张量信息。")
                all_reshape_valid = False

    return all_reshape_valid

def export_to_onnx(model, dummy_input, onnx_path, input_names, output_names, dynamic_axes):
    model.eval()
    
    # 强制指定输入输出的静态维度
    torch.onnx.export(
        model=model,
        args=dummy_input,
        f=onnx_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=16,
        do_constant_folding=True,
        training=torch.onnx.TrainingMode.EVAL,
        # 添加以下参数强制固定非批次维度
        keep_initializers_as_inputs=True,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX
    )
    
    # 形状推断
    onnx_model = onnx.load(onnx_path)
    inferred_model = shape_inference.infer_shapes(onnx_model)
    onnx.save(inferred_model, onnx_path)
    print(f"✅ ONNX 模型已成功导出: {onnx_path}")

    # 打印输入和输出的形状
    for input_info in onnx_model.graph.input:
        print(f"Input name: {input_info.name}, shape: {[dim.dim_value for dim in input_info.type.tensor_type.shape.dim]}")
    for output_info in onnx_model.graph.output:
        print(f"Output name: {output_info.name}, shape: {[dim.dim_value for dim in output_info.type.tensor_type.shape.dim]}")

    # 检查 Reshape 节点
    if not check_reshape_nodes(inferred_model):
        print("❌ 检测到 Reshape 节点问题，请检查模型。")
        return False
    return True

def validate_onnx_gpu(onnx_path, dummy_input_shape):
    options = SessionOptions()
    
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    
    content = torch.randn(*dummy_input_shape).cpu().numpy()
    style = torch.randn(*dummy_input_shape).cpu().numpy()
    ort_inputs = {"content": content, "style": style}
    
    try:
        session = InferenceSession(onnx_path, providers=providers)
        ort_outputs = session.run(None, ort_inputs)
        print(f"✅ ONNX GPU 推理通过，输出形状: {ort_outputs[0].shape}")
        return True
    except Exception as e:
        print(f"❌ ONNX GPU 推理失败: {str(e)}")
        return False

def main(checkpoint_path, output_path, batch_size=1):
    if not validate_cuda_availability():
        return
    
    from model import StyleTransfer
    from hyperparam import Hyperparameter

    hyper_param = Hyperparameter()
    hyper_param.image_shape = [256, 256]  
    model = StyleTransfer(
        image_shape=tuple(hyper_param.image_shape),
        style_dim=hyper_param.style_dim,
        style_kernel=hyper_param.style_kernel,
        groups=hyper_param.groups,
        fixed_batch_size=batch_size  # 确保传递固定批次大小
    ).cuda()

    checkpoint = torch.load(checkpoint_path, map_location="cuda")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 确保输入尺寸严格固定为256x256
    dummy_content = torch.randn(batch_size, 3, 256, 256, device="cuda")
    dummy_style = torch.randn(batch_size, 3, 256, 256, device="cuda")

    # 仅允许批次维度动态
    dynamic_axes = {
        "content": {0: "batch_size"},
        "style": {0: "batch_size"},
        "output": {0: "batch_size"},
    }

    if export_to_onnx(
        model=model,
        dummy_input=(dummy_content, dummy_style),
        onnx_path=output_path,
        input_names=["content", "style"],
        output_names=["output"],
        dynamic_axes=dynamic_axes
    ):
        validate_onnx_gpu(output_path, dummy_input_shape=(batch_size, 3, 256, 256))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="模型权重路径 (.pt)")
    parser.add_argument("--output", type=str, required=True, help="ONNX 模型输出路径")
    parser.add_argument("--batch-size", type=int, default=1, help="ONNX 批大小（默认 1）")
    args = parser.parse_args()
    
    try:
        main(args.checkpoint, args.output, args.batch_size)
    except Exception as e:
        print(f"❌ 发生错误: {str(e)}")