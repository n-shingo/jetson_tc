# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 14:03:42 2017

@author: shingo

つくばチャレンジプログラム用ツール群
"""

import time
from time import sleep


# 周波数を計算するクラス
class FrequencyChecker:

    # コンストラクタ（この時点で計測開始）    
    def __init__(self, buffsize=10):
        self._buffsize = buffsize
        self._data = [time.time()]
    
    # リセット（最初から計測やりなおし）
    def reset(self, buffsize=10):
        self._buffsize = buffsize
        self._data = [time.time()]
                      
    # 周波数を計算する
    def frequency(self):
        
        # バッファいっぱいであれば最初を削る
        if len(self._data) == self._buffsize:
            self._data = self._data[1:]

        # 現在の時刻を追加
        self._data.append( time.time() )
        
        # 周波数を計算
        dt = (self._data[-1] - self._data[0])
        return (len(self._data)-1)/dt



# 時間を計測するだけのクラス
class Timer():
    def __init_(self):
        self._start = time.time()

    def start(self):
        self._start = time.time()
        return self
                       
    def time(self):
        return time.time() - self._start
        

# Test program
if __name__ == '__main__':
    
    chcker = FrequencyChecker()
    timer = Timer()
    
    while(True):
        
        timer.start()
        sleep(0.5)  #2Hz
        print( '{0:.2f} Hz,  {1:.2f} s'
              .format(chcker.frequency(), timer.time()) )




