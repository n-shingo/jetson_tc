#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 14:50:59 2017

@author: shingo
"""


import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
import time

# LeNet5の改良
# サイズが元と違うだけ
class MyLeNet5( chainer.Chain ):

    insize = 256

    def __init__(self):
        super(MyLeNet5, self).__init__(
            conv1 = L.Convolution2D(3, 6, 5, stride=1 ),
            conv2 = L.Convolution2D(6, 16, 5, stride=1 ),
            conv3 = L.Convolution2D(16, 120, 5, stride=1 ),
            fc4 = L.Linear(None, 84),
            fc5 = L.Linear(84,2),
        )

    def __call__(self, x):
        h = F.relu( self.conv1(x) )
        h = F.max_pooling_2d(h, 2, 2)
        h = F.relu( self.conv2(h) )
        h = F.max_pooling_2d(h, 2, 2)
        h = F.relu( self.conv3(h) )
        h = F.relu( self.fc4(h) )
        return self.fc5(h)

# LeNet5の改良モデル 
# Dropout と BatchNormalizationが追加されている
# Drop out率をコンストラクタで設定できる

class MyLeNet5WithDo( chainer.Chain ):

    def __init__(self, output, do_ratio=0.3):
        super(MyLeNet5WithDo, self).__init__(
            conv1 = L.Convolution2D(3, 12, 5, stride=1 ),
            b1 = L.BatchNormalization( 12 ),
            conv2 = L.Convolution2D(12, 32, 5, stride=1 ),
            b2 = L.BatchNormalization( 32 ),
            conv3 = L.Convolution2D(32, 240, 5, stride=1 ),
            b3 = L.BatchNormalization( 240 ),
            fc4 = L.Linear(None, 168),
            fc5 = L.Linear(168, output)
        )
        self._train = True
        self.dropout = do_ratio 

    def __call__(self, x ):
        h = self.conv1(x)
        h = self.b1( h, test = not self._train )
        h = F.relu( h )
        h = F.dropout( h, ratio = self.dropout, train=self._train )
        h = F.max_pooling_2d(h, 2, 2)
        h = F.dropout( h, ratio = self.dropout, train=self._train )
        
        h = self.conv2(h)
        h = self.b2( h, test = not self._train )
        h = F.relu( h )
        h = F.dropout( h, ratio = self.dropout, train=self._train )
        h = F.max_pooling_2d( h, 2, 2 )
        h = F.dropout( h, ratio = self.dropout, train=self._train )

        h = self.conv3(h)
        h = self.b3( h, test = not self._train )
        h = F.relu( h )
        h = F.dropout( h, ratio = self.dropout, train=self._train )

        h = self.fc4(h)
        h = F.relu( h )
        h = F.dropout( h, ratio = self.dropout, train=self._train )
        
        return self.fc5(h)

    @property
    def train(self):
        return self._train

    @train.setter
    def train( self, val ):
        self._train = val




# LeNet5モデル
# 出力は Classifierで 1 of 10 で使用する
class LeNet5(chainer.Chain):
    
    '''
    参考:http://docs.chainer.org/en/stable/tutorial/convnet.html
    '''
    
    insize = 32
    
    def __init__(self):
        super(LeNet5, self).__init__(
            conv1 = L.Convolution2D(
                in_channels=3, out_channels=6, ksize=5, stride=1),
            conv2 = L.Convolution2D(
                in_channels=6, out_channels=16, ksize=5, stride=1),
            conv3 = L.Convolution2D(
                in_channels=16, out_channels=120, ksize=4, stride=1),
            fc4 = L.Linear(None, 84),
            fc5 = L.Linear(84, 10),
        )
        self.train = True

    def __call__(self, x):
        h = F.sigmoid(self.conv1(x))
        h = F.max_pooling_2d(h, 2, 2)
        h = F.sigmoid(self.conv2(h))
        h = F.max_pooling_2d(h, 2, 2)
        h = F.sigmoid(self.conv3(h))
        h = F.sigmoid(self.fc4(h))
        return self.fc5(h)

        
class MLP(chainer.Chain):

    def __init__(self):
        super(MLP, self).__init__(
            # the size of the inputs to each layer will be inferred
            l1 = L.Linear(None, 1000),  # n_in -> n_units
            l2 = L.Linear(None, 1000),  # n_units -> n_units
            l3 = L.Linear(None, 2),  # n_units -> n_out
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)

if __name__ == '__main__':

    # Calculate LeNet5 forward
    lenet5 = LeNet5()
    x = np.random.rand(1, 1, 32, 32).astype(np.float32)
    y = lenet5(x)
    
    # Calculate MyNet forward
    mynet = MyLeNet5()
    x = np.random.rand(1, 3, 32, 32).astype(np.float32)
    start_time = time.time()
    y = mynet(x)
    print( time.time()-start_time, 'MyLeNet5' )
    
    # Calculate LeNet5 forward
    lenet5 = LeNet5()
    x = np.random.rand(1, 1, 32, 32).astype(np.float32)
    start_time = time.time()
    y = lenet5(x)
    print( time.time()-start_time, 'LeNet5')
   
    
