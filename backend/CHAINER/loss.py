r'''
basic wrapper for loss function
'''
from . import chainer

softmax_cross_entropy = chainer.functions.softmax_cross_entropy