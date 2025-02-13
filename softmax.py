import numpy as np

def softmax(x):
    result = np.zeros_like(x)

    sum = 0

    for i in range(x.shape[0]):
        exp = np.exp(x[i])
        sum += exp

    for i in range(x.shape[0]):
        exp = np.exp(x[i])
        result[i] = (exp/sum)

    return result


#TEST

# x = np.array([[-0.21635479],
#               [-0.41350353],
#               [-0.20350754],
#               [ 0.34841453]], dtype=float)
#
#
# rizz = softmax(x)
# print(rizz)
