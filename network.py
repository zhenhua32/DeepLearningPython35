# %load network.py

"""
network.py
~~~~~~~~~~
IT WORKS

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np

class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        # size 是一个list结构, 定义各层的神经元数量
        # biases 的个数和输出层的神经元的个数有关, 比如一开始输入是2, 输出是3, 个数就是三个
        # weights 的位置和输出层输入层有关, 而且是输出层在前, 输入层在后
        # 比如 [00] 表示输入层的第一个和输出层的第一个相连
        # [01] 表示输入层的第二个和输出层的第一个相连
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]  # np.random.randn 返回标准正态分布
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        # 前反馈网络, 对于给定的a, 返回当前的输出
        for b, w in zip(self.biases, self.weights):  # 对于输入层, 迭代得到输出
            a = sigmoid(np.dot(w, a)+b)  # 为什么用a, 因为下一层的输入就是这一层的输出
        return a  # 最后输出 a 的形状和最后一层神经元有关, 应该是 (最后一层的个数, 1) 的数组

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        # 随机梯度下降方法
        # training_data 是类似于(x, y)的训练数据, x 是输入, y是期望输出
        # epochs 是时代, 纪元的意思, 训练的次数
        # mini_batch_size 是最小批次
        # eta 是学习速率
        # test_data 格式和 train_data 一样
        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):
            random.shuffle(training_data)  # 随机打乱数组
            mini_batches = [
                training_data[k:k+mini_batch_size]  # 将整个 train_data 划分成 mini_batch_size 大小
                for k in range(0, n, mini_batch_size)]  # 列表推导式写成三行了
            for mini_batch in mini_batches: # 从上面可以看出, 所有的数据都会被用在一个 epoch 中
                self.update_mini_batch(mini_batch, eta)  # 见下面的函数定义, 核心
            if test_data and n_test:
                print("Epoch {} : {} / {}".format(j,self.evaluate(test_data),n_test))  # 验证
            # else:  # 为什么放else, 去掉就能在每次都显示了
            print("Epoch {} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        # 更新 weights 和 biases
        nabla_b = [np.zeros(b.shape) for b in self.biases]  # 复制形状, 用零填充
        nabla_w = [np.zeros(w.shape) for w in self.weights]  # nabla 是微分符号的意思, 倒三角形
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)  # 又见新函数, 核心中的核心
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        # 这里是 步骤3, 更新梯度
        self.weights = [w-(eta/len(mini_batch))*nw  # 通过 nabla_w 重新计算 weights
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        # 反向传播
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            # 步骤2.1
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        # 步骤2.2, 计算最后一个输出误差, delta
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        # 这里是 BP3
        nabla_b[-1] = delta
        # 这里是 BP4
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())  # transpose() 是矩阵转置
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        # 步骤2.3, 误差反向传播, 依次计算 第L-1层到第2层
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            # 右边的 delta 是 l+1 层的 delta
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)  # np.argmax 返回最大值的索引
                        for (x, y) in test_data]  # 为什么用索引, 因为数字识别时输出是一个(10, 1)的十维数组
        return sum(int(x == y) for (x, y) in test_results) # y 是正确值, 也是索引

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        # 代价函数的导数, 当前的代价函数是平方代价函数
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))


if __name__ == '__main__':
    net = Network([2,3,1])
    print(net.biases)
    print(net.weights)
    # for x in net.biases:
    #    print(x)
    random_a = np.random.randn(4, 1)
    print(random_a)
    # result = net.feedforward(random_a)
    # print(result)
    # print(np.argmax(result))
