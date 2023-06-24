import numpy


def activation(x):
    return 1 / (1 + numpy.exp(-x))


def sigma_derivative(x):
    return x * (1 - x)


def neuralnet():
    # params -------------------------------------------------------
    # input
    obj_dim = 3
    #body
    h1_dim = 4
    h2_dim = 5
    #output
    out_dim = 1
    # x = [count_object, count_signs]
    x = numpy.array([[0, 1, 0],
                  [1, 0, 0],
                  [1, 1, 1]])
    # y = [count_objects, 1]
    y = numpy.array([[1],
                  [1],
                  [0]])
    # result
    global z

    # weight matrices [left_neuron, right_neuron]
    w1 = 2 * numpy.random.random((obj_dim, h1_dim)) - 1
    w2 = 2 * numpy.random.random((h1_dim, h2_dim)) - 1
    w3 = 2 * numpy.random.random((h2_dim, out_dim)) - 1
    # processing
    iterations_num = 2
    antgr_speed = 1.1

    # iterations -------------------------------------------------------
    for i in range(iterations_num):
        # forward propagation
            # First level - X
            # Second level
        t1 = x @ w1
        h1 = activation(t1)
            # Third level
        t2 = h1 @ w2
        h2 = activation(t2)
            # Forth (last) level (without activation)
        z = h2 @ w3

        # backward propagation
            # Clear z error
        e_full = y - z
            # Forth level local error gradient
        sigma_z = e_full * sigma_derivative(z)
            # Clear h2 error
        e_h2 = sigma_z @ w3.T
            # Third level local error gradient
        sigma_h2 = e_h2 * sigma_derivative(h2)
            # Clear h1 error
        e_h1 = sigma_h2 @ w2.T
            # Second level local error gradient
        sigma_h1 = e_h1 * sigma_derivative(h1)
            # Update weights
        w3 += antgr_speed * h2.T @ sigma_z
        w2 += antgr_speed * h1.T @ sigma_h2
        w1 += antgr_speed * x.T @ sigma_h1

    print(z)


if __name__ == '__main__':
    neuralnet()