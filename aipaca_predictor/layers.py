from typing import Dict
from typing import Any
from tensorflow.python.keras.layers.core import Dense
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.layers.pooling import MaxPooling2D


class PoolLayer(object):
    def __init__(self, layer: MaxPooling2D, input_size: int, channels_in: int):
        self.type: str = layer.pool_function.__name__
        # Assume strides are symmetric
        self.strides: int = layer.strides[0]
        self.pool_size: int = layer.pool_size[0]
        self.padding: str = layer.padding
        self.channels_in: int = channels_in
        self.channels_out: int = channels_in
        self.output_size: int = _get_conv_output_size(
            input_size=input_size,
            kernel_size=self.pool_size,
            padding=self.padding,
            strides=self.strides,
        )


class ConvLayer(object):
    def __init__(self, layer: Conv2D, input_size: int, channels_in: int):
        self.type: str = "Convolution"
        self.mat_size: float = float(input_size)
        self.kernel_size: int = layer.kernel_size[0]
        self.channels_in: int = channels_in
        self.channels_out: int = layer.kernel.shape[-1]
        self.strides: int = layer.strides[0]
        self.padding: str = layer.padding
        self.use_bias: bool = layer.use_bias
        self.activation = _get_activation_name(layer.activation)
        self.output_size: int = _get_conv_output_size(
            input_size=input_size,
            kernel_size=self.kernel_size,
            padding=self.padding,
            strides=self.strides,
        )


class DenseLayer(object):
    def __init__(self, layer: Dense, input_size: int, channels_in: int):
        # Assume d
        self.type: str = "Convolution"
        self.mat_size: float = float(input_size) if input_size else 1
        self.kernel_size: int = input_size if input_size else 1
        self.channels_in: int = channels_in
        self.channels_out: int = layer.units
        self.strides: int = 1
        # Because output_size always 1, thus same iff input_size is also 1
        self.padding: str = "same" if input_size == 1 else "valid"
        self.use_bias: bool = True
        self.activation = _get_activation_name(layer.activation)
        self.output_size: int = 1


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
