import onnx

model = onnx.load("analog.onnx")
onnx.checker.check_model(model)