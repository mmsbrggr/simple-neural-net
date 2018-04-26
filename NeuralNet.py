import numpy as np
from InputLayer import InputLayer
from HiddenLayer import HiddenLayer
from OutputLayer import OutputLayer
from Connection import Connection


# The actual Neural Net which constructs
# all the layers and connections between them
class NeuralNet:

    @staticmethod
    def get_errors(outputs, expected_outputs):
        return expected_outputs - outputs

    def __init__(self, number_inputs, hidden_sizes, number_outputs, labels):
        self.input_layer = None
        self.output_layer = None
        self.labels = labels
        self.initialize_layers(number_inputs, hidden_sizes, number_outputs)

    # Set up the recursive data structure according to the passed parameters
    def initialize_layers(self, number_inputs, hidden_sizes, number_outputs):
        self.input_layer = InputLayer("Input", number_inputs)
        from_layer = self.input_layer

        count = 1
        for size in hidden_sizes:
            to_layer = HiddenLayer("Hidden " + str(count), size)
            connection = Connection(from_layer, to_layer)
            to_layer.set_input_connection(connection)
            from_layer.set_output_connection(connection)
            from_layer = to_layer
            count += 1

        self.output_layer = OutputLayer("Output", number_outputs)
        connection = Connection(from_layer, self.output_layer)
        from_layer.set_output_connection(connection)
        self.output_layer.set_input_connection(connection)

    def predict(self, inputs):
        outputs = self.feed_forward(np.array(inputs))
        return self.labels[np.argmax(outputs)]

    # Feed the passed input parameters to the input layer and return
    # the result from the output layer
    def feed_forward(self, inputs):
        self.input_layer.feed_forward(np.array(inputs))
        return self.output_layer.get_outputs()

    # Trains the network via the backpropagation algorithm
    def train(self, inputs, targets):
        outputs = self.feed_forward(np.array(inputs))
        errors = self.get_errors(outputs, np.array(targets))
        self.output_layer.back_propagate(errors)
