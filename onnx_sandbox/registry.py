import onnxruntime
import onnxscript
import torch
from onnxscript import opset18


class Model(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.ops.aten.add(x, y)


input_x = torch.randn(3, 4)
input_y = torch.randn(3, 4)
model = Model()

# Now we declare a ONNX Script function that implements `aten::add.Tensor`.
# The function name (e.g. `custom_aten_add`) is displayed in the ONNX graph.
custom_aten = onnxscript.values.Opset(domain="custom.aten", version=1)


@onnxscript.script(custom_aten)
def custom_aten_add(input_x, input_y, alpha: float = 1.0):
    alpha = opset18.CastLike(alpha, input_y)
    input_y = opset18.Mul(input_y, alpha)
    return opset18.Add(input_x, input_y)


onnx_registry = torch.onnx.OnnxRegistry()
onnx_registry.register_op(
    namespace="aten", op_name="add", overload="Tensor", function=custom_aten_add
)

print(
    f"aten::add.Tensor is supported by ONNX registry: \
      {onnx_registry.is_registered_op(namespace='aten', op_name='add', overload='Tensor')}"
)
export_options = torch.onnx.ExportOptions(onnx_registry=onnx_registry)
onnx_program = torch.onnx.dynamo_export(model, input_x, input_y, export_options=export_options)
