# YOLOX-TensorRT in C++

## Envirenment

Telsa = T4


CUDA = 11.1


TensorRT = 8.0.3.4

## Step 1: pth2onnx

1. get source code of yolox

```
git clone https://github.com/Megvii-BaseDetection/YOLOX.git
cd YOLOX
```

2. modify the code

```


# line 206 forward fuction in yolox/models/yolo_head.py. Replace the commented code with the uncommented code
# self.hw = [x.shape[-2:] for x in outputs] 
self.hw = [list(map(int, x.shape[-2:])) for x in outputs]


# line 208 forward function in yolox/models/yolo_head.py. Replace the commented code with the uncommented code
# [batch, n_anchors_all, 85]
# outputs = torch.cat(
#     [x.flatten(start_dim=2) for x in outputs], dim=2
# ).permute(0, 2, 1)
proc_view = lambda x: x.view(-1, int(x.size(1)), int(x.size(2) * x.size(3)))
outputs = torch.cat(
    [proc_view(x) for x in outputs], dim=2
).permute(0, 2, 1)


# line 253 decode_output function in yolox/models/yolo_head.py Replace the commented code with the uncommented code
#outputs[..., :2] = (outputs[..., :2] + grids) * strides
#outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
#return outputs
xy = (outputs[..., :2] + grids) * strides
wh = torch.exp(outputs[..., 2:4]) * strides
return torch.cat((xy, wh, outputs[..., 4:]), dim=-1
```

3. download pth model and convert to onnx

```

# download model
wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.pth

# export
export PYTHONPATH=$PYTHONPATH:.
python tools/export_onnx.py -c yolox_m.pth -f exps/default/yolox_m.py --output-name=yolox_m.onnx --dynamic --no-onnxsim

```

## Step 2: onnx2engine

```
cd /path/to/TensorRT-8.0.3.4/bin
./trtexec --onnx=/path/to/model.onnx --minShapes=input:1x3x640x640 --optShapes=input:4x3x640x640 --maxShapes=input:16x3x640x640  --verbose --avgRuns=10 --plugins --saveEngine=/path/to/model.engine
```

## Step 3: make this project

First you should set the TensorRT path and CUDA path in CMakeLists.txt.

compile

```
git clone git@github.com:LIUHAO121/YOLOX_TensorRT_cpp.git
cd YOLOX_TensorRT_cpp
mkdir -p build
cd build
cmake ..
make
cd ..
```

run

```
./build/yolox /path/to/model.engine -i input.jpg 
```
