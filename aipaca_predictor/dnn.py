class DNN(dict):
    """Class for deep neural network architecture"""

    def __init__(self, input_dimension, input_size):
        self["layers"] = {}
        self["input"] = {}
        self["input"]["dimension"] = input_dimension
        self["input"]["size"] = input_size

    def add_layer(self, layer_type, layer_name, **kwargs):
        """Adds a layer to the class instance
        Args:
            layer_type: Type of layer ('Convolution', 'Fully_connected' or
                    'Max_pool')
            layer_name: Name of layer (string)
        Layer type specific args:
            Convolution:
                kernelsize
                channels_out
                padding
                strides
                use_bias
                activation
            Max_pool:
                pool_size
                strides
                padding
            Fully_connected:
        """

        num_layers = len(self["layers"])
        new_layer = num_layers + 1

        self["layers"][new_layer] = {}  # Create new layer
        self["layers"][new_layer]["name"] = layer_name
        self["layers"][new_layer]["type"] = layer_type

        if layer_type.lower() == "convolution":
            self["layers"][new_layer]["matsize"] = kwargs["mat_size"]
            self["layers"][new_layer]["kernelsize"] = kwargs["kernel_size"]
            self["layers"][new_layer]["channels_in"] = kwargs["channels_in"]
            self["layers"][new_layer]["channels_out"] = kwargs["channels_out"]
            self["layers"][new_layer]["padding"] = kwargs["padding"]
            self["layers"][new_layer]["strides"] = kwargs["strides"]
            self["layers"][new_layer]["use_bias"] = kwargs["use_bias"]
            self["layers"][new_layer]["activation"] = kwargs["activation"]
            self["layers"][new_layer]["output_size"] = kwargs["output_size"]

        if layer_type.lower() == "max_pool":
            self["layers"][new_layer]["pool_size"] = kwargs["pool_size"]
            self["layers"][new_layer]["strides"] = kwargs["strides"]
            self["layers"][new_layer]["padding"] = kwargs["padding"]
            self["layers"][new_layer]["output_size"] = kwargs["output_size"]
            self["layers"][new_layer]["channels_out"] = kwargs["channels_out"]

        output_size = self["layers"][new_layer]["output_size"]
        print(
            "%s (%s), now %dx%d with %d channels"
            % (
                layer_name,
                layer_type,
                output_size,
                output_size,
                self["layers"][new_layer]["channels_out"],
            )
        )
