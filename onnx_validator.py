import onnx
from onnx import shape_inference
from onnxruntime import InferenceSession, SessionOptions

import torch
import onnxruntime

import numpy as np
import warnings
import os
import sys
from collections import defaultdict


# 忽略OnnxRuntime的警告信息
warnings.filterwarnings("ignore", category=FutureWarning, module="onnxruntime")


class ONNXValidator:
    """ONNX模型验证器类"""
    
    def __init__(self, onnx_path, verbose=True):
        """
        初始化ONNX验证器
        
        参数:
            onnx_path (str): ONNX模型文件路径
            verbose (bool): 是否显示详细日志
        """
        self.onnx_path = onnx_path
        self.verbose = verbose
        self.model = None
        self.session = None
        self.providers = None
        self.input_names = None
        self.output_names = None
        self.issues_found = []
        
        # 检查模型文件是否存在
        if not os.path.exists(onnx_path):
            self._log_error(f"模型文件不存在: {onnx_path}")
            sys.exit(1)
    
    def _log_info(self, message):
        """输出信息日志"""
        if self.verbose:
            print(f"ℹ️ {message}")
    
    def _log_success(self, message):
        """输出成功日志"""
        print(f"✅ {message}")
    
    def _log_warning(self, message):
        """输出警告日志"""
        print(f"⚠️ {message}")
        self.issues_found.append(("警告", message))
    
    def _log_error(self, message):
        """输出错误日志"""
        print(f"❌ {message}")
        self.issues_found.append(("错误", message))
    
    def _log_section(self, title):
        """输出分隔线和章节标题"""
        print(f"\n{'='*50}")
        print(f"【{title}】")
        print(f"{'-'*50}")
    
    def load_model(self):
        """加载ONNX模型"""
        try:
            self._log_info(f"正在加载模型: {self.onnx_path}")
            self.model = onnx.load(self.onnx_path)
            self._log_success(f"成功加载ONNX模型")
            return True
        except Exception as e:
            self._log_error(f"加载ONNX模型失败: {str(e)}")
            return False
    
    def check_model_metadata(self):
        """检查模型元数据"""
        self._log_section("模型元数据检查")
        
        if not self.model:
            self._log_error("模型未加载，无法检查元数据")
            return False
        
        # 检查模型版本
        self._log_info(f"ONNX IR版本: {self.model.ir_version}")
        self._log_info(f"生产者名称: {self.model.producer_name if self.model.producer_name else '未指定'}")
        self._log_info(f"生产者版本: {self.model.producer_version if self.model.producer_version else '未指定'}")
        self._log_info(f"域: {self.model.domain if self.model.domain else '未指定'}")
        self._log_info(f"模型版本: {self.model.model_version if self.model.model_version else '未指定'}")
        
        # 检查操作集版本
        opset_imports = [f"{opset.domain} (版本 {opset.version})" for opset in self.model.opset_import]
        self._log_info(f"操作集导入: {', '.join(opset_imports)}")
        
        # 检查图名称
        self._log_info(f"图名称: {self.model.graph.name if self.model.graph.name else '未命名'}")
        
        # 获取输入输出信息
        inputs = [f"{i.name} ({self._get_tensor_shape_str(i)})" for i in self.model.graph.input]
        outputs = [f"{o.name} ({self._get_tensor_shape_str(o)})" for o in self.model.graph.output]
        
        self._log_info(f"模型输入 ({len(inputs)}): {', '.join(inputs)}")
        self._log_info(f"模型输出 ({len(outputs)}): {', '.join(outputs)}")
        
        # 设置输入输出名称
        self.input_names = [i.name for i in self.model.graph.input]
        self.output_names = [o.name for o in self.model.graph.output]
        
        return True
    
    def _get_tensor_shape_str(self, tensor):
        """获取张量形状的字符串表示"""
        shape = tensor.type.tensor_type.shape
        dims = []
        for dim in shape.dim:
            if dim.HasField("dim_value"):
                dims.append(str(dim.dim_value))
            elif dim.HasField("dim_param"):
                dims.append(dim.dim_param)
            else:
                dims.append("?")
        return f"{tensor.type.tensor_type.elem_type}, [{', '.join(dims)}]"
    
    def check_graph_structure(self):
        """检查图结构"""
        self._log_section("图结构检查")
        
        if not self.model:
            self._log_error("模型未加载，无法检查图结构")
            return False
        
        # 节点统计
        node_types = defaultdict(int)
        total_nodes = len(self.model.graph.node)
        
        for node in self.model.graph.node:
            node_types[node.op_type] += 1
        
        self._log_info(f"总节点数: {total_nodes}")
        self._log_info("操作类型统计:")
        for op_type, count in sorted(node_types.items(), key=lambda x: x[1], reverse=True):
            self._log_info(f"  - {op_type}: {count} ({count/total_nodes*100:.1f}%)")
        
        # 检查是否有孤立节点
        all_inputs = set()
        all_outputs = set()
        
        for node in self.model.graph.node:
            for input_name in node.input:
                if input_name:  # 跳过空输入
                    all_inputs.add(input_name)
            for output_name in node.output:
                if output_name:  # 跳过空输出
                    all_outputs.add(output_name)
        
        # 检查每个节点的输出是否被其他节点使用
        unused_outputs = all_outputs - all_inputs - set(self.output_names)
        if unused_outputs:
            self._log_warning(f"发现 {len(unused_outputs)} 个未使用的中间输出，可能存在死代码")
            if self.verbose:
                for unused in list(unused_outputs)[:5]:  # 只显示前5个
                    self._log_info(f"  - 未使用的输出: {unused}")
                if len(unused_outputs) > 5:
                    self._log_info(f"  - ... 还有 {len(unused_outputs) - 5} 个")
        
        # 检查输入节点和最终输出
        missing_inputs = set(self.input_names) - all_inputs
        if missing_inputs:
            self._log_error(f"模型定义了输入但未被使用: {missing_inputs}")
        
        missing_outputs = set(self.output_names) - all_outputs
        if missing_outputs:
            self._log_error(f"模型定义了输出但未被生成: {missing_outputs}")
        
        return True
    
    def check_reshape_nodes(self):
        """检查所有Reshape节点"""
        self._log_section("Reshape节点检查")
        
        if not self.model:
            self._log_error("模型未加载，无法检查Reshape节点")
            return False
        
        reshape_nodes = [node for node in self.model.graph.node if node.op_type == 'Reshape']
        if not reshape_nodes:
            self._log_info("模型中没有Reshape节点")
            return True
        
        self._log_info(f"找到 {len(reshape_nodes)} 个Reshape节点")
        
        # 节点名称到输入/输出类型映射
        tensor_info = {}
        
        # 初始化输入张量信息
        for tensor in self.model.graph.input:
            tensor_info[tensor.name] = tensor
        
        # 初始化中间张量信息
        for tensor in self.model.graph.value_info:
            tensor_info[tensor.name] = tensor
        
        # 初始化输出张量信息
        for tensor in self.model.graph.output:
            tensor_info[tensor.name] = tensor
        
        # 记录有问题的reshape节点
        problematic_nodes = []
        
        # 检查每个Reshape节点
        for node_index, node in enumerate(reshape_nodes):
            self._log_info(f"\n检查第 {node_index+1}/{len(reshape_nodes)} 个Reshape节点: {node.name}")
            
            # 获取输入输出名称
            input_name = node.input[0] if len(node.input) > 0 else None
            shape_name = node.input[1] if len(node.input) > 1 else None
            output_name = node.output[0] if len(node.output) > 0 else None
            
            if not (input_name and output_name):
                self._log_error(f"节点 {node.name} 输入或输出缺失")
                problematic_nodes.append(node.name)
                continue
            
            # 获取输入输出张量信息
            input_tensor = tensor_info.get(input_name)
            output_tensor = tensor_info.get(output_name)
            
            if not input_tensor:
                self._log_warning(f"无法找到输入张量 {input_name} 的信息")
            
            if not output_tensor:
                self._log_warning(f"无法找到输出张量 {output_name} 的信息")
            
            if not (input_tensor and output_tensor):
                self._log_warning(f"由于信息不足，跳过节点 {node.name} 的详细检查")
                continue
            
            # 分析张量形状
            input_dims = self._parse_tensor_shape(input_tensor)
            output_dims = self._parse_tensor_shape(output_tensor)
            
            self._log_info("输入形状: " + self._format_dims(input_dims))
            self._log_info("输出形状: " + self._format_dims(output_dims))
            
            # 检查reshape操作是否合法
            is_valid, reason = self._validate_reshape(input_dims, output_dims)
            
            if is_valid:
                self._log_success(f"Reshape节点 {node.name} 验证通过")
            else:
                self._log_error(f"Reshape节点 {node.name} 验证失败: {reason}")
                problematic_nodes.append(node.name)
                
            # 检查shape张量是否为常量
            if shape_name:
                is_const_shape = False
                for init in self.model.graph.initializer:
                    if init.name == shape_name:
                        is_const_shape = True
                        shape_data = self._get_initializer_data(init)
                        self._log_info(f"Reshape使用常量形状: {shape_data}")
                        break
                
                if not is_const_shape:
                    self._log_warning(f"Reshape使用动态形状张量: {shape_name} (可能会影响某些运行时)")
        
        if problematic_nodes:
            self._log_error(f"发现 {len(problematic_nodes)} 个有问题的Reshape节点: {', '.join(problematic_nodes[:5])}")
            return False
        else:
            self._log_success("所有Reshape节点验证通过")
            return True
    
    def _get_initializer_data(self, initializer):
        """从初始化器中提取数据"""
        if initializer.data_type == 7:  # INT64
            return np.frombuffer(initializer.raw_data, dtype=np.int64).tolist()
        elif initializer.data_type == 1:  # FLOAT
            return np.frombuffer(initializer.raw_data, dtype=np.float32).tolist()
        else:
            return f"[未支持的数据类型: {initializer.data_type}]"
    
    def _parse_tensor_shape(self, tensor):
        """解析张量形状"""
        dims = []
        for i, dim in enumerate(tensor.type.tensor_type.shape.dim):
            if dim.HasField('dim_value'):
                value = dim.dim_value
                status = "静态" if value > 0 else "动态(0)" if value == 0 else "负值"
            elif dim.HasField('dim_param'):
                value = dim.dim_param  
                status = "符号"
            else:
                value = None
                status = "未知"
            dims.append((i, value, status))
        return dims
    
    def _format_dims(self, dims):
        """格式化维度信息"""
        result = []
        for idx, val, status in dims:
            result.append(f"{val}({status})")
        return "[" + ", ".join(result) + "]"
    
    def _validate_reshape(self, input_dims, output_dims):
        """验证Reshape操作的有效性"""
        # 计算元素数量
        input_calculable, input_size, input_dynamic = self._calculate_size(input_dims)
        output_calculable, output_size, output_dynamic = self._calculate_size(output_dims)
        
        # 对于可计算的静态形状，比较元素总数
        if input_calculable and output_calculable:
            if input_size != output_size:
                return False, f"静态形状元素数量不匹配: 输入{input_size} ≠ 输出{output_size}"
            return True, "静态形状验证通过"
        
        # 对于特殊的-1维度(自动推断)，暂时认为有效
        has_special_dim = any(v == -1 for _, v, _ in output_dims if isinstance(v, int))
        if has_special_dim:
            return True, "包含自动推断维度(-1)，运行时验证"
        
        # 对于动态或符号形状，无法确定是否有效，但提供警告
        if not input_calculable or not output_calculable:
            return True, "包含动态/符号维度，需要运行时验证"
        
        return True, "形状验证通过"
    
    def _calculate_size(self, dim_info):
        """计算元素总数"""
        size = 1
        dynamic_dims = []
        has_symbolic = False
        
        for idx, val, status in dim_info:
            if status == "静态" and isinstance(val, int) and val > 0:
                size *= val
            elif status == "动态(0)" or val == 0:
                dynamic_dims.append(idx)
            elif status == "符号" or not isinstance(val, int):
                has_symbolic = True
            elif status == "负值" and val == -1:  # 特殊的-1维度(自动推断)
                dynamic_dims.append(idx)
        
        return (not dynamic_dims and not has_symbolic), size, dynamic_dims
    
    def check_cuda_availability(self):
        """检查CUDA可用性"""
        self._log_section("CUDA可用性检查")
        
        if not torch.cuda.is_available():
            self._log_warning("CUDA不可用，将使用CPU模式运行")
            return False
        
        device_name = torch.cuda.get_device_name(0)
        device_count = torch.cuda.device_count()
        
        self._log_success(f"CUDA可用，检测到 {device_count} 个设备")
        self._log_info(f"当前设备: {device_name}")
        
        # 检查 CUDA 内存
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        self._log_info(f"显存总量: {total_mem:.2f} GB")
        
        return True
    
    def check_onnx_runtime(self):
        """检查OnnxRuntime环境"""
        self._log_section("OnnxRuntime检查")
        
        # 检查OnnxRuntime版本
        self._log_info(f"OnnxRuntime版本: {onnxruntime.__version__}")
        
        # 获取可用的执行提供者
        available_providers = onnxruntime.get_available_providers()
        self._log_info(f"可用的执行提供者: {available_providers}")
        
        # 检查是否支持CUDA
        if "CUDAExecutionProvider" in available_providers:
            self._log_success("OnnxRuntime支持CUDA加速")
            self.providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            self._log_warning("OnnxRuntime不支持CUDA加速，将使用CPU")
            self.providers = ["CPUExecutionProvider"]
        
        return True
    
    def create_inference_session(self):
        """创建推理会话"""
        self._log_section("创建推理会话")
        
        if not self.providers:
            self.check_onnx_runtime()
        
        options = SessionOptions()
        # 设置会话选项，如图优化级别、线程数等
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = 0  # 使用默认线程数
        
        try:
            self._log_info(f"正在创建推理会话，使用提供者: {self.providers}")
            self.session = InferenceSession(
                self.onnx_path, 
                providers=self.providers,
                sess_options=options
            )
            
            # 获取模型输入信息
            model_inputs = self.session.get_inputs()
            model_outputs = self.session.get_outputs()
            
            self._log_info("模型输入详情:")
            for i, input_info in enumerate(model_inputs):
                self._log_info(f"  - 输入 {i}: 名称={input_info.name}, 形状={input_info.shape}, 类型={input_info.type}")
            
            self._log_info("模型输出详情:")
            for i, output_info in enumerate(model_outputs):
                self._log_info(f"  - 输出 {i}: 名称={output_info.name}, 形状={output_info.shape}, 类型={output_info.type}")
            
            self._log_success("成功创建推理会话")
            return True
        
        except Exception as e:
            self._log_error(f"创建推理会话失败: {str(e)}")
            return False
    
    def run_inference_test(self, dummy_input_shapes):
        """运行推理测试"""
        self._log_section("推理测试")
        
        if not self.session:
            if not self.create_inference_session():
                return False
        
        # 验证输入形状字典的完整性
        model_inputs = self.session.get_inputs()
        for input_info in model_inputs:
            if input_info.name not in dummy_input_shapes:
                self._log_error(f"缺少输入 {input_info.name} 的形状")
                self._log_info(f"请为所有输入提供形状: {[i.name for i in model_inputs]}")
                return False
        
        # 创建随机输入数据
        ort_inputs = {}
        for input_info in model_inputs:
            input_name = input_info.name
            input_shape = dummy_input_shapes[input_name]
            
            # 检查输入形状是否匹配
            if len(input_shape) != len(input_info.shape):
                self._log_warning(f"输入 {input_name} 的维度数量不匹配: 预期 {len(input_info.shape)}, 实际 {len(input_shape)}")
            
            # 检查数据类型
            if 'float' in input_info.type:
                data_type = np.float32
            elif 'int64' in input_info.type:
                data_type = np.int64
            elif 'int32' in input_info.type:
                data_type = np.int32
            else:
                self._log_warning(f"未知的输入类型 {input_info.type}，将使用float32")
                data_type = np.float32
            
            # 生成随机数据
            ort_inputs[input_name] = np.random.randn(*input_shape).astype(data_type)
        
        # 记录推理时间
        import time
        start_time = time.time()
        
        try:
            ort_outputs = self.session.run(None, ort_inputs)
            end_time = time.time()
            
            # 输出推理结果信息
            self._log_success(f"推理成功，耗时: {(end_time-start_time)*1000:.2f} ms")
            
            for i, output in enumerate(ort_outputs):
                self._log_info(f"输出 {i} 形状: {output.shape}")
                
                # 检查输出是否包含NaN或Inf
                has_nan = np.isnan(output).any()
                has_inf = np.isinf(output).any()
                
                if has_nan:
                    self._log_warning(f"输出 {i} 包含NaN值")
                
                if has_inf:
                    self._log_warning(f"输出 {i} 包含Inf值")
                
                # 输出一些基本统计信息
                if output.size > 0:
                    self._log_info(f"  - 最小值: {output.min()}")
                    self._log_info(f"  - 最大值: {output.max()}")
                    self._log_info(f"  - 均值: {output.mean()}")
                    self._log_info(f"  - 标准差: {output.std()}")
            
            return True
        
        except Exception as e:
            self._log_error(f"推理测试失败: {str(e)}")
            return False
    
    def check_model_integrity(self):
        """检查模型完整性"""
        self._log_section("模型完整性检查")
        
        if not self.model:
            if not self.load_model():
                return False
        
        try:
            # 使用ONNX内置的检查器验证模型
            onnx.checker.check_model(self.model)
            self._log_success("ONNX模型结构验证通过")
        except Exception as e:
            self._log_error(f"ONNX模型结构验证失败: {str(e)}")
            return False
        
        try:
            # 尝试进行形状推断
            inferred_model = shape_inference.infer_shapes(self.model)
            self._log_success("ONNX形状推断成功")
            
            # 比较原始模型和推断后模型的value_info数量
            original_value_info = len(self.model.graph.value_info)
            inferred_value_info = len(inferred_model.graph.value_info)
            
            if inferred_value_info > original_value_info:
                self._log_info(f"形状推断添加了 {inferred_value_info - original_value_info} 个中间张量形状信息")
            
        except Exception as e:
            self._log_warning(f"ONNX形状推断失败: {str(e)}")
        
        return True
    
    def full_validation(self, dummy_input_shapes):
        """执行完整验证"""
        self._log_section("开始完整验证")
        
        # 重置问题列表
        self.issues_found = []
        
        # 1. 加载模型
        if not self.load_model():
            self._log_error("验证终止: 无法加载模型")
            return False
        
        # 2. 检查模型完整性
        self.check_model_integrity()
        
        # 3. 检查模型元数据
        self.check_model_metadata()
        
        # 4. 检查图结构
        self.check_graph_structure()
        
        # 5. 检查Reshape节点
        self.check_reshape_nodes()
        
        # 6. 检查CUDA可用性
        has_cuda = self.check_cuda_availability()
        
        # 7. 检查OnnxRuntime
        self.check_onnx_runtime()
        
        # 8. 运行推理测试
        inference_ok = self.run_inference_test(dummy_input_shapes)
        
        # 汇总结果
        self._log_section("验证结果汇总")
        
        if self.issues_found:
            self._log_warning(f"发现 {len(self.issues_found)} 个问题:")
            
            # 按类型分组显示问题
            errors = [issue for issue in self.issues_found if issue[0] == "错误"]
            warnings = [issue for issue in self.issues_found if issue[0] == "警告"]
            
            if errors:
                self._log_error(f"错误 ({len(errors)}):")
                for _, message in errors:
                    print(f"  - {message}")
            
            if warnings:
                self._log_warning(f"警告 ({len(warnings)}):")
                for _, message in warnings:
                    print(f"  - {message}")
        else:
            self._log_success("未发现任何问题")
        
        # 最终结果
        if inference_ok and not any(issue[0] == "错误" for issue in self.issues_found):
            self._log_success("模型验证通过，可以用于生产环境")
            return True
        else:
            self._log_warning("模型验证未完全通过，请解决上述问题后再用于生产环境")
            return False
        
        return True


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description="ONNX模型验证工具")
    parser.add_argument("--onnx", type=str, required=True, help="ONNX模型文件路径")
    parser.add_argument("--input-shape", type=str, nargs="+", required=True, 
                        help="输入张量形状，格式: <name>:dim1,dim2,... 例如: input:1,3,224,224")
    parser.add_argument("--verbose", "-v", action="store_true", help="显示详细日志")
    
    args = parser.parse_args()
    
    try:
        # 解析输入形状参数
        dummy_input_shapes = {}
        for shape_str in args.input_shape:
            parts = shape_str.split(":")
            if len(parts) != 2:
                print(f"❌ 输入形状格式错误: {shape_str}")
                print("正确格式: <name>:dim1,dim2,... 例如: input:1,3,224,224")
                return 1
            
            name = parts[0]
            dims = [int(d) for d in parts[1].split(",")]
            dummy_input_shapes[name] = dims
        
        # 创建验证器并运行完整验证
        validator = ONNXValidator(args.onnx, verbose=args.verbose)
        validator.full_validation(dummy_input_shapes)
        
    except Exception as e:
        print(f"❌ 发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())