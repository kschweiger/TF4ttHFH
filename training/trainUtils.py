import numpy as np
from math import sqrt 

from keras import backend as K


# def correntropyLoss(sigma = 0.2):
#     """
#     Implementation of the correntropy 
#     """
#     def Ksig(alpha):
#         Keras.exp(-Keras.square(alpha)/(2*sigma**2))
    
#     def robustKernel(alpha):
#         return (1.0/(sqrt(2 * np.pi) * sigma))* Ksig(alpha)

#     def loss(y_pred, y_true):
#         return -Keras.sum(robust_kernel(y_pred - y_true))

#     return loss


def r_square(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return ( 1 - SS_res/(SS_tot + K.epsilon()))
