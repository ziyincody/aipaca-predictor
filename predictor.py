from dnn import DNN
from typing import Dict
from typing import Any
import requests
import json


class PoolLayer(object):
    def __init__(self, layer: Dict[str, Any], input_size: int, channels_in: int):
        self.type: str = "Max_pool"
        # Assume strides are symmetric
        self.strides: int = layer["strides"][0]
        self.pool_size: int = layer["pool_size"][0]
        self.padding: str = layer["padding"]
        self.channels_in: int = channels_in
        self.channels_out: int = channels_in
        self.output_size: int = _get_conv_output_size(
            input_size=input_size,
            kernel_size=self.pool_size,
            padding=self.padding,
            strides=self.strides,
        )


class ConvLayer(object):
    def __init__(self, layer: Dict[str, Any], input_size: int, channels_in: int):
        self.type: str = "Convolution"
        self.mat_size: float = float(input_size)
        self.kernel_size: int = layer["kernel_size"][0]
        self.channels_in: int = channels_in
        self.channels_out: int = layer["kernel"].shape[-1]
        self.strides: int = layer["strides"][0]
        self.padding: int = layer["padding"]
        self.use_bias: bool = layer["use_bias"]
        self.activation = _get_activation_name(layer["activation"])
        self.output_size: int = _get_conv_output_size(
            input_size=input_size,
            kernel_size=self.kernel_size,
            padding=self.padding,
            strides=self.strides,
        )


class DenseLayer(object):
    def __init__(self, layer: Dict[str, Any], input_size: int, channels_in: int):
        self.type: str = "Convolution"
        self.mat_size: float = float(input_size)
        self.kernel_size: int = input_size
        self.channels_in: int = channels_in
        self.channels_out: int = layer["units"]
        self.strides: int = 1
        self.padding: int = "same" if input_size == 1 else "valid"
        self.use_bias: bool = True
        self.activation = _get_activation_name(layer["activation"])
        self.output_size: int = 1


def to_predict(model, batch_size: int, gpu_name: str, optimizer: str = "adam") -> int:
    """
    Given a keras model and hardware specs, output the estimated training time
    """
    dnn = parse_cnn(model.layers)

    data = {
        "dnn": dnn,
        "batch_size": batch_size,
        "optimizer": optimizer,
        "gpu_name": gpu_name,
    }

    url = "http://127.0.0.1:8000/predict"

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


def parse_dnn(model_layers):
    pass


def _get_activation_name(activation_func):
    return activation_func.__wrapped__._keras_api_names[0].split(".")[-1]


def _get_conv_output_size(
    input_size: int, kernel_size: int, padding: str, strides: int
):
    """
    (W - F + 2P) / S + 1
    """
    if padding == "valid":
        padding_size = 0
    else:
        padding_size = ((strides - 1) * input_size - strides + kernel_size) / 2

    return (input_size - kernel_size + 2 * padding_size) / strides + 1
