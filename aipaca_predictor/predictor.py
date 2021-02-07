from aipaca_predictor.dnn import DNN
from aipaca_predictor.layers import PoolLayer
from aipaca_predictor.layers import ConvLayer
from aipaca_predictor.layers import DenseLayer
import requests
import json
from typing import List
from typing import Optional
from tensorflow.python.client import device_lib
from tensorflow.core.framework.device_attributes_pb2 import DeviceAttributes


SUPPORT_GPU_TYPES = {"1080Ti", "K40", "K80", "M60", "P100", "T4", "V100"}


def to_predict(model, batch_size: int, optimizer: str = "adam") -> int:
    """
    Given a keras model and hardware specs, output the estimated training time
    """
    gpu_name = _find_gpu_type()
    if not gpu_name:
        print(f"Your GPU is not supported. not one of {SUPPORT_GPU_TYPES}")
        return

    print(f"Detected that you have {gpu_name}")

    dnn = parse_cnn(model.layers)

    data = {
        "dnn": dnn,
        "batch_size": batch_size,
        "optimizer": optimizer,
        "gpu_name": gpu_name,
    }

    url = "http://35.182.126.77:8000/predict"

    resp = requests.post(url, data=json.dumps(data))

    print(resp.json())


def parse_cnn(model_layers):
    input_shape = vars(model_layers[0])["_batch_input_shape"]
    dimension = input_shape[-1]
    size = input_shape[1]
    previous_channel = dimension
    previous_output_size = size

    dnn = DNN(dimension, size)
    for layer in model_layers[1:]:
        layer_class_name = layer.__class__.__name__
        layer_name = layer.name
        layer_dict = vars(layer)
        if layer_class_name == "MaxPooling2D":
            layer = PoolLayer(
                layer=layer_dict,
                input_size=previous_output_size,
                channels_in=previous_channel,
            )
        elif layer_class_name == "Conv2D":
            layer = ConvLayer(
                layer=layer_dict,
                input_size=previous_output_size,
                channels_in=previous_channel,
            )
        elif layer_class_name == "Flatten":
            continue
        elif layer_class_name == "Dense":
            layer = DenseLayer(
                layer=layer_dict,
                input_size=previous_output_size,
                channels_in=previous_channel,
            )
        previous_channel = layer.channels_out
        previous_output_size = layer.output_size
        dnn.add_layer(layer_name=layer_name, layer_type=layer.type, **vars(layer))

    return dnn


def _find_gpu_type() -> Optional[str]:
    local_device_protos: List[DeviceAttributes] = device_lib.list_local_devices()
    for device in local_device_protos:
        for gpu_type in SUPPORT_GPU_TYPES:
            if gpu_type.upper() in device.physical_device_desc.upper():
                return gpu_type
    return None
