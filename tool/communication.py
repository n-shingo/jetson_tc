# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 13:01:35 2017

@author: shingo

PC と通信するためのクラス群
"""

import socket
import sys

# PCからの通信データを解析するクラス
class CommandParser:
    
    # コンストラクタ
    def __init__(self, jetson_status = 0):
        self.JetsonStatus = jetson_status
        self._data = ''

    # 通信データを解析する
    def parse(self, rcvData ):
        
        # データがなければ何もせず、終了
        if len(rcvData) == 0 :
            return
            
        # データを一連に連結
        line = ''
        for data in rcvData:
            line = line + data.decode("UTF-8")
            
        # 逆順にする
        line = line[::-1]

        # 有効な値を探す
        for c in line:
            if c=='1':
                self.JetsonStatus = 1
            elif c=='2':
                self.JetsonStatus = 2
            elif c=='3':
                self.JetsonStatus = 3

        
        
    

# 送信データ
class SendData:
    
    def __init__(self):
        self.JetsonStatus = 0
        self.ThetaSStatus = 0
        self.WebcamStatus = 0
        self.PersonResult = 0
        self.PersonPos = 0
        self.SignalResult = 0

    # 通信用文字列データを返す        
    def comStr(self):
        s = "S,{0},{1},{2},{3},{4},{5},E,"\
            .format(
                    self.JetsonStatus,
                    self.ThetaSStatus,
                    self.WebcamStatus,
                    self.PersonResult,
                    self.PersonPos,
                    self.SignalResult,
                    )
            
        return s
        

# UDP 送信クラス
class UDPSender:
    
    # コンストラクタ
    def __init__(self, rcv_ip, rcv_port, snd_ip, snd_port ):
        
        self._rcv_ip = rcv_ip
        self._rcv_port = rcv_port
        self._snd_ip = snd_ip
        self._snd_port = snd_port
        self._sock = socket.socket( socket.AF_INET, socket.SOCK_DGRAM )
        self._sock.bind( (snd_ip, snd_port) )
        
    def __del__(self):
        self._sock.close()
        
    # メッセージを送る
    def sendMessage( self, msg ):
        assert type(msg) == str, 'str型しかsendMessage@UDPできません'
        message = msg.encode('utf-8')
        self._sock.sendto(message, (self._rcv_ip, self._rcv_port) )

    def close(self):
        self._sock.close()

        
# UDP 受信クラス
class UDPRecver:
    
    #コンストラクタ
    def __init__( self, rcv_ip, rcv_port ):
        self._rcv_ip = rcv_ip
        self._rcv_port = rcv_port
        self._bufsize = 4096
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM )
        self._sock.bind( (rcv_ip, rcv_port) )
        
    def __del__(self):
        self._sock.close()
        
    def receiveMessage( self, timeout = 0.05 ):
        all_data = []
        self._sock.settimeout(timeout)
        while True:
            data = None
            try:
                data, client = self._sock.recvfrom(self._bufsize)
                all_data.append(data)
            except socket.timeout:
                pass
            
            if data == None:
                break
        
        return all_data
                
        
    def close(self):
        self._sock.close()


# 自分のIPを取得するクラス( localhost や '127.0.0.1'などではなく)
def get_ownIP():
    # https://qiita.com/kjunichi/items/8e4967d04c3a1f6af35e
    ip = [(s.connect(('8.8.8.8', 80)), s.getsockname()[0], s.close()) for s in [socket.socket(socket.AF_INET, socket.SOCK_DGRAM)]][0][1]
    return ip



# UDPテストプログラム
if __name__ == '__main__':
    
    # pythonのメジャーバージョン
    py_ver = sys.version_info.major

    # 自分自身のIP取得 (自分に送る)
    ip = get_ownIP()
    print( 'this host ip is {0}'.format(ip) )
    
    # 通信ホスト＆ポート番号
    rcv_ip = ip
    rcv_port = 51103
    snd_ip = ip
    snd_port = 50000
    
    #ソケット作成
    sender = UDPSender( rcv_ip, rcv_port, snd_ip, snd_port)
    rcver = UDPRecver( rcv_ip, rcv_port )
    
    while True:
        if py_ver == 3:
            data = input('Data > ')
        else:
            data = raw_input('Data >')

        if not data:
            break
        sender.sendMessage( str(data) )
        data = rcver.receiveMessage()
        if len(data)>0 :
            print( data )
        
    sender.close()
    rcver.close()
