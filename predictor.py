from dnn import DNN
from layers import PoolLayer
from layers import ConvLayer
from layers import DenseLayer
import requests
import json


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
