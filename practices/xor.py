# -*- coding: utf-8 -*-


import numpy as np


def lmap(fn, array):
    res = []
    for a in array:
        res.append(lmap(fn, a) if isinstance(a, np.ndarray) else fn(a))
    return np.array(res)


class ReluActivator(object):
    def forward(self, matrix):
        return lmap(lambda a: max(0, a), matrix)

    def backward(self, matrix):
        return lmap(lambda a: (1 if a > 0 else 0), matrix)


class IdentityActivator(object):
    def forward(self, x):
        return lmap(lambda a: (1 if a > 0 else 0), x)

    def backward(self, x):
        return lmap(lambda _: 0, x)


class Layer(object):
    def __init__(self, input_size, output_size, activator):
        self.input_size = input_size
        self.output_size = output_size
        self.activator = activator
        # self.W = np.ones((input_size, output_size))
        self.W = np.random.uniform(-1, 1, (input_size, output_size))
        self.b = np.random.uniform(-1, 1, (1, output_size))
        self.output = np.zeros((1, output_size))

    def forward(self, input_array):
        self.input = input_array
        self.output = self.activator.forward(
            np.dot(input_array, self.W) + self.b)

    def backward(self, delta_array):
        print('delta: %s' % delta_array)
        self.delta = self.activator.backward(self.input).T * np.dot(
            self.W, delta_array)
        self.W_grad = np.dot(delta_array, self.input).T
        self.b_grad = delta_array.T

    def update(self, learning_rate):
        self.W += learning_rate * self.W_grad
        self.b += learning_rate * self.b_grad

    def dump(self):
        print('W: %s\nb: %s' % (self.W, self.b))


class Network(object):
    def __init__(self, layers, activators):
        self.layers = []
        for i in range(len(layers) - 1):
            self.layers.append(Layer(layers[i], layers[i + 1], activators[i]))

    def predict(self, sample):
        output = np.array([sample])
        for layer in self.layers:
            layer.forward(output)
            output = layer.output
        return output

    def train(self, labels, data_set, rate, epoch):
        for i in range(epoch):
            for d in range(len(data_set)):
                self.train_one_sample(labels[d], data_set[d], rate)

    def train_one_sample(self, label, sample, rate):
        self.predict(sample)
        self.calc_gradient(label)
        self.update_weight(rate)

    def calc_gradient(self, label):
        delta = self.layers[-1].activator.backward(self.layers[-1].output) * (
            label - self.layers[-1].output)
        for layer in self.layers[::-1]:
            layer.backward(delta)
            delta = layer.delta
        return delta

    def update_weight(self, rate):
        for layer in self.layers:
            layer.update(rate)

    def dump(self):
        for layer in self.layers:
            layer.dump()

    def loss(self, output, label):
        return 0.5 * ((label - output) ** 2).sum()


def evaluate(network, test_data_set, test_labels):
    error = 0
    total = len(test_data_set)
    print(network.dump())
    for i in range(total):
        print(network.predict(test_data_set[i]))
        print(test_labels[i])
        if test_labels[i] != network.predict(test_data_set[i]):
            error += 1
    return float(error) / float(total)


if __name__ == '__main__':
    data_set = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    labels = np.array([0, 1, 1, 0])
    print('--> Load data successed!')

    identityActivator = IdentityActivator()
    reluActivator = ReluActivator()
    network = Network(
        [2, 2, 1],
        [reluActivator, identityActivator]
    )
    network.dump()
    print('--> Init network successed!')

    epoch = 0
    last_error_ratio = 1.0
    while True:
        epoch += 1
        print('--> epoch %d start.' % epoch)

        network.train(labels, data_set, 0.01, 1)
        print('--> epoch %d finished, loss %f' % (
            epoch, network.loss(labels[-1], network.predict(data_set[-1]))))

        if epoch % 2 == 0:
            error_rate = evaluate(network, data_set, labels)
            print('--> after epoch %d, error ratio is %f'
                  % (epoch, error_rate))

            if error_rate > last_error_ratio:
                break
            else:
                last_error_ratio = error_rate
