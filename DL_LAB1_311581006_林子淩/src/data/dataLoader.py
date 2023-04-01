import numpy as np
import matplotlib.pyplot as plt

def load_data(data_type):
    """data loader"""
    data_type = data_type.lower()
    if data_type == "linear":
        return generate_linear()
    elif data_type == "xor":
        return generate_XOR_easy()
    else:
        raise ValueError("incorrect generate data type, needs to be 'linear' or 'xor'")



def generate_linear(n=100):
    """generate linear inputs"""
    pts = np.random.uniform(0, 1, (n,2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0],pt[1]])
        distance = (pt[0]-pt[1])/1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n,1)

def generate_XOR_easy():
    """generate XOR inputs"""
    inputs = []
    labels = []

    for i in range(11):
        inputs.append([0.1 * i, 0.1 * i])
        labels.append(0)

        if 0.1 * i == 0.5:
            continue

        inputs.append([0.1 * i, 1 - 0.1 * i])
        labels.append(1)

    return np.array(inputs), np.array(labels).reshape(21, 1)

def plot_data(type, x, y):
    plt.title(type, fontsize=18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    plt.show()

if __name__=="__main__":
    x_linear, y_linear = generate_linear()
    x_xor, y_xor = generate_XOR_easy()
    plot_data('linear',x_linear, y_linear)
    plot_data('xor',x_xor, y_xor)