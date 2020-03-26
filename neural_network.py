import numpy as np


def sigmoid(z, derv=False):
    if derv:
        return z * (1 - z)
    return 1 / (1 + np.exp(-z))


class NeuralNetwork:
    input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], int)
    output_data = np.array([[0], [1], [1], [0]], int)

    def __init__(self):
        self.input = np.array([0, 0])
        self.weights1 = np.random.rand(4, 2)
        self.ideal = 0
        self.weights2 = np.random.rand(1, 4)
        self.output = np.zeros(1)

    def feedforward(self):
        self.layer = sigmoid(np.dot(self.input, self.weights1.T))
        self.output = sigmoid(np.dot(self.layer, self.weights2.T))

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        delta2 = 2 * (self.ideal - self.output) * sigmoid(self.output, derv=True)
        delta1 = np.dot(self.weights2.T, delta2) * sigmoid(self.layer, derv=True)
        d_weights2 = np.multiply(delta2, self.layer.T)
        d_weights1 = np.dot(delta1.reshape(4, 1), self.input.reshape(1, 2))
        # update the weights with the derivative (slope) of the loss function
        self.weights2 += d_weights2
        self.weights1 += d_weights1

    def think(self, inp):
        probably = sigmoid(np.dot(sigmoid(np.dot(inp, self.weights1.T)), self.weights2.T))[0]
        return 1 if probably >= 0.5 else 0, probably

    def train(self, quantity_to_view: int = 100, *, number_of_epoch: int):
        for epoch in range(1, number_of_epoch+1):
            if epoch % quantity_to_view == 0:
                print(f'EPOCH {epoch}')
                for data in self.input_data:
                    print(f'{data} --> {self.think(data)}')
            for inp, outp in zip(self.input_data, self.output_data):
                self.input = inp
                self.ideal = outp
                self.feedforward()
                self.backprop()

    def __repr__(self):
        return f'\nWEIGHTS1 \n{self.weights1}\n' \
               f'WEIGHTS2 \n{self.weights2}\n'


if __name__ == '__main__':
    xor = NeuralNetwork()
    print(xor)
    xor.train(number_of_epoch=10000)
    print(xor)