import argparse
from onnx import ModelProto
import tensorrt as trt
from pathlib import Path

def build_engine(TRT_LOGGER , onnx_path, shape , half):

    """
    This is the function to create the TensorRT engine
    Args:
        onnx_path : Path to onnx_file.
        shape : Shape of the input of the ONNX file.
    """
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1) as network, builder.create_builder_config() as config, trt.OnnxParser(network, TRT_LOGGER) as parser:
        config.max_workspace_size = (1024 << 20)
        with open(onnx_path, 'rb') as model:
            parser.parse(model.read())
        network.get_input(0).shape = shape
        if half:
            config.set_flag(trt.BuilderFlag.FP16)
        
        engine = builder.build_engine(network, config)
        return engine

def save_engine(engine, file_name):
    buf = engine.serialize()
    with open(file_name, 'wb') as f:
        f.write(buf)

def onnx2engine(onnx_file , engine_file , half):

    print(f"start {onnx_file} => {engine_file}")
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    
    batch_size = 1

    model = ModelProto()
    with open(onnx_file, "rb") as f:
        model.ParseFromString(f.read())

    d0 = model.graph.input[0].type.tensor_type.shape.dim[1].dim_value
    d1 = model.graph.input[0].type.tensor_type.shape.dim[2].dim_value
    d2 = model.graph.input[0].type.tensor_type.shape.dim[3].dim_value
    shape = [batch_size , d0, d1 ,d2]
    engine = build_engine(TRT_LOGGER , onnx_file, shape , half)
    save_engine(engine, engine_file)
    print(f"input shape : {shape}" )

    output = []
    for idx in range(len(model.graph.output[0].type.tensor_type.shape.dim)):
        output.append(model.graph.output[0].type.tensor_type.shape.dim[idx].dim_value)

    # output_0 = model.graph.output[0].type.tensor_type.shape.dim[0].dim_value
    # output_1 = model.graph.output[0].type.tensor_type.shape.dim[1].dim_value
    # output_2 = model.graph.output[0].type.tensor_type.shape.dim[2].dim_value
    # model_output_shape = (output_0 , output_1 , output_2)

    print(f"output shape : {output}" )

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', type=str, help='onnx model path')
    parser.add_argument('--half', action="store_true", help='fp16')

    opt = parser.parse_args()
    return opt

def main(opt):
    print(opt)

    opt['pt'] =  opt['onnx'].replace(".onnx" , ".pt")
    half = opt['half']

    onnx2engine(opt['pt'].replace(".pt" , ".onnx") , opt['pt'].replace(".pt" , ".engine") , half)
        

if __name__ == '__main__':
    opt = parse_opt()
    main(vars(opt))