# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 11:30:28 2017

@author: shingo

Jetson program for Tsukuba Challenge
"""

# PCとJetsonのIPアドレス
PC_IP = "172.29.46.41" 
JETSON_IP = "172.29.46.39"


import sys
import cv2
import numpy as np
from chainer import serializers, cuda, Variable
import chainer.links as L
from time import sleep
from tool.communication import UDPSender, UDPRecver, get_ownIP
from tool.communication import SendData, CommandParser
from tool.tools import FrequencyChecker, Timer
from tool.people_searcher import PeopleSearcher
from models.LeNet5 import MyLeNet5WithDo



""" カメラ関連 """
CAM_IDs = [0,1,2]
THETAS_SIZE = (720,1280,3)
WEBCAM_SIZE = (480,640,3)
CAPTURE_TIMEOUT_THRESHOLD = 0.5 # キャプチャがタイムアウトしたか判定する時間[sec]


""" 通信関連 """
# JETSON_IP  # 必要なし自動取得
JETSON_SND_PORT = 54112
JETSON_RCV_PORT = 54113
PC_SND_PORT = "Any Ports" # 使用していない
PC_RCV_PORT = 64113
RCV_TIMEOUT = 0.001   # Jetsonの受信時のタイムアウト時間[sec]


""" 人探し関連 """
GPU = 0  # 0:GPU, -1:CPU
BE_NN_FILE = 'result/mylenet5_100epoc_thetas_without_green3.npz'  # 人学習済みNNファイル
PEROSON_THRESHOLD = 0.90  # 人居る／居ないの閾値
SHOW_PROCESS_IMGS = True

def main():
    
    jetsonStatus = 1 # 0:sick, 1:idling, 2:searching person, 3:detecting signal
 
    #------------------------ カメラ準備 -----------------------------#
    print( "\n[Preparing to capture from USB cameras]" )

    thetasCamID = -1
    webcamCamID = -1
    thetasCap = None
    webcamCap = None
    
    # ThetaS と Webcam の ID を調べて設定
    for camid in CAM_IDs:
        try:
            cap = cv2.VideoCapture( camid )
        except:
            continue
        
        ret, img = cap.read()
        
        if( ret==True and img.shape == THETAS_SIZE ):
            thetasCap = cap
            thetasCamID = camid
        elif( ret==True and img.shape == WEBCAM_SIZE ):
            webcamCap = cap
            webcamCamID = camid
        else:
            cap.release()
            
    # linux(jetson)上でカメラが落ちたか判断するための画像
    last_thetas_img = None
    last_webcam_img = None
            
    print( "#---------------------------------------------------\n#" );
    print( "# ThetaS Camera ID : {0}".format( thetasCamID ))
    print( "# Webcam Camera ID : {0}".format( webcamCamID ))
    print( "#\n#----------------------------------------------[done]\n" );



    #------------------------ 通信準備 -----------------------------#
    print( "\n[Preparing communication with PC]")

    jetsonIP = JETSON_IP # get_ownIP()
    jetsonSndPort = JETSON_SND_PORT
    jetsonRcvPort = JETSON_RCV_PORT
    
    pcIP = PC_IP
    pcSndPort = PC_SND_PORT
    pcRcvPort = PC_RCV_PORT

    # UDP socket の準備
    sender = UDPSender( pcIP, pcRcvPort, jetsonIP, jetsonSndPort )
    recver = UDPRecver( jetsonIP, jetsonRcvPort )

    print( "#---------------------------------------------------\n#" );
    print( "# Jetson({0}:{1}) -----> PC({2}:{3})"
          .format( jetsonIP, jetsonSndPort, pcIP, pcRcvPort) )
    print( "# Jetson({2}:{3}) <----- PC({0}:{1})"
          .format( pcIP, pcSndPort, jetsonIP, jetsonRcvPort) )
    print( "#\n#----------------------------------------------[done]\n" );
    
    
    
    # ---------------------- 人探し準備 --------------------------- #
    print( "\n[Preparig searching target people]" )
    print( "#---------------------------------------------------\n#" );

    if GPU >= 0:
        print( "# GPU will be used")
    else:
        print( "# GPU will not be used" )

    # 人検知の NN読込
    model_object = MyLeNet5WithDo(output=2, do_ratio=0.3)
    model_being = L.Classifier(model_object)
    serializers.load_npz( BE_NN_FILE, model_being)

    if( GPU>=0 ):
        model_being.to_gpu(GPU)
    print( "# CNN has been loaded from npz file" )
    
    # 人検索クラス
    print( "# Preparing PeopleSearcher module..." )
    searcher = PeopleSearcher(model_being, gpu=GPU, threshold=PEROSON_THRESHOLD)
    # 初回の探索は遅いので、一回だけ無駄探索しておく
    tmp_img = np.zeros( THETAS_SIZE, dtype=np.uint8 )
    searcher.find_people( tmp_img, show_imgs=False )
    print( "#\n#----------------------------------------------[done]\n" );
    


     

    """ ############################################# """
    """                 処理ループ開始                """
    """ ############################################# """
    print( "\n************** Start loop process *******************\n" )
    freqChck = FrequencyChecker() # 周波数チェッカ
    cmdParser = CommandParser(jetson_status=1)
    while( True ):

        
        """------------------- PCからのデータ受信 -----------------"""
        # 通信データ取得　
        rcvData = recver.receiveMessage( timeout=RCV_TIMEOUT )
        cmdParser.parse(rcvData)
        jetsonStatus = cmdParser.JetsonStatus

            
        """----------------- PCへの送信データの準備 -----------------"""
        data = SendData()
        data.JetsonStatus = jetsonStatus

        
        
        """---------------------カメラ画像取得 ---------------------"""
        # THETA S
        thetasImg = None
        if thetasCap is not None:

            thetasImg = get_capture_image(thetasCap)
            
            if( thetasImg is not None and last_thetas_img is not None) :
                if( np.allclose( thetasImg[:64,:64,0], last_thetas_img) ):
                    thetasImg = None
                    
            if thetasImg is not None:
                last_thetas_img = thetasImg[:64,:64,0]

            # 画像が取れているかチェック
            if( thetasImg is None ):
                thetasCap.release()
                thetasCap = None
                data.ThetaSStatus = 0
            else:
                data.ThetaSStatus = 1
                
        
        # WEB CAM
        webcamImg = None
        if webcamCap is not None:

            webcamImg = get_capture_image(webcamCap)
            
            if( webcamImg is not None and last_webcam_img is not None) :
                if( np.allclose( webcamImg[:64,:64,0], last_webcam_img) ):
                    thetasImg = None
                    
            if webcamImg is not None:
                last_webcam_img = webcamImg[:64,:64,0]

            # 画像が取れているかチェック
            if( webcamImg is None ):
                webcamCap.release()
                webcamCap = None
                data.WebcamStatus = 0
            else:
                data.WebcamStatus = 1
            
                
        # カメラが２台とも落ちていたら、安全のため 30Hz程度で回す
        if( thetasCap == None and webcamCap == None ):
            sleep(1.0/30.0)
            
            
        """---------------- 人探し ----------------------"""
        if( thetasImg is not None and jetsonStatus == 2):
            img, pos = searcher.find_people( thetasImg, show_imgs=SHOW_PROCESS_IMGS )
            
            if pos is not None : # found!
                data.PersonResult = 1
                data.PersonPos = pos
            
            else: # not found
                data.PersonResult = 2
        
        
        """---------------- 交通信号検出 --------------------"""
        # TODO
        
        
        
        """------------- PC へデータ送信 ------------------"""
        sender.sendMessage( data.comStr() )

        
        
        """---------------- 結果表示 --------------------"""
        # Output result
        sys.stdout.write( "\x1b[KLoop frequency: {0:.2f}Hz\n".format( freqChck.frequency() ) );
        print( "[Jetson Status]" )
        sys.stdout.write('\x1b[K' )
        print( '  Jtsn:{0}, ThtaS:{1}, WebCam:{2}, Psn:{3}, PsnPos:{4:.2f}, Sign:{5}'
              .format(jetsonStatus, data.ThetaSStatus, data.WebcamStatus,
                  data.PersonResult, data.PersonPos, data.SignalResult ) )
        sys.stdout.write('\x1b[3A')

        
    # 処理ループ終了

# キャプチャから画像を取得する関数
def get_capture_image( capture ):
    
    # キャプチャが初期化されていなければ終了
    if capture is None:
        return None
    
    # 画像取得
    timer = Timer().start()
    ret, img = capture.read()
    time_to_get = timer.time()

    # 取得失敗
    if ret == False:
        return None
    # 時間がかかっていたら、おそらく落ちている
    elif time_to_get > CAPTURE_TIMEOUT_THRESHOLD:
        return None
    # 成功
    else:
        return img
    
 
# main関数の実行
if __name__ == '__main__':
    sys.stdout.write('\x1b[2J\x1b[0;0H')
    print( "" )
    print( "***********************************************************")
    print( "*                                                         *")
    print( "*          Tsukuba Challeng 2017 Jetson Program           *")
    print( "*                                                         *")
    print( "***********************************************************")
    print( "" )
    print( "This program is executed by using below modules" )
    print( "Python: {0}".format(sys.version))
    print( "OpenCV: {0}".format(cv2.__version__))
    print( "" )
    print( "" )
    print( "Starting program..." )
    main()