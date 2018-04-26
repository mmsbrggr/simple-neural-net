from Layer import Layer


class InputLayer(Layer):

    def feed_forward(self, inputs):
        self.last_inputs = inputs
        self.last_outputs = inputs
        self.output_connection.feed_forward(inputs)
