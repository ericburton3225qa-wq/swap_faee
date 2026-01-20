import onnx
from onnxconverter_common import float16

model = onnx.load("inswapper_128.onnx")
model_fp16 = float16.convert_float_to_float16(model)
onnx.save(model_fp16, "inswapper_128_fp16.onnx")