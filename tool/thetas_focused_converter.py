# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 18:34:02 2017

thieta S画像(1280x720)を
正距円筒図に変換する関数
正し、必要なところだけ抜き出す特別バージョン

@author: shingo
"""

import numpy as np;
import cv2;

def main():
    img = cv2.imread( 'test_thetasimg.png' )
    conv = ThetaSFocusedConverter()
    dst = conv.convert(img)
    
    cv2.imshow( 'image', dst )
    cv2.waitKey()
    
    cv2.destroyAllWindows()


class ThetaSFocusedConverter:
    def __init__(self, blobsize=64, hcnt = 4, wcnt = 8, imgline=[2]):
        
        """ 正距離円筒図において...
            blocksize: 単位ブロックのサイズ[pix]
            hcnt:高さのブロック数 (つまり高さは blobsize x hcnt [pix])
            wcnt:幅のブロック数 (つまり高さは blobsize x wcnt [pix])
            imgline: 生成する画像のline番号(tupleで指定)
        """
        
        #カメラパラメータ初期値(SN:170159向け)
        self._focus = 2.0
        self._radius = 284.0
        self._center_f = (319.0, 319.0)
        self._center_b = (959.0, 319.0)
        self._dangle = 0.0 # -0.004  # ２つの魚眼レンズのずれ角度

        # 正距円筒図パラメータ
        w, h = blobsize*wcnt, blobsize*hcnt
        self._equirect_size = (w, h)
        self._equirect_shift_angle_deg = 90 # 正面画像左端の位置
        
        # 返す画像に関するパラメータ
        self._blobsize = blobsize
        self._hcnt = hcnt
        self._wcnt = wcnt
        self._imgline = imgline
        
        # remap用テーブルの作成と取得
        self._mapu, self._mapv = self._makeEquirectangleMap()
        

    # ThetaS画像を正距円筒図に変換する        
    def convert(self, src):
        w, h = self._equirect_size
        dst = cv2.remap( src, self._mapu, self._mapv, cv2.INTER_LINEAR )
        return dst
        
    # 変換マップを作成する
    def _makeEquirectangleMap(self):

        # 全体の正距円筒図のサイズ        
        w, h = self._equirect_size

        # shift量をpixelに変換
        shift = int(w * self._equirect_shift_angle_deg / 360.0+0.5)

        # マップの作成        
        mapu = np.zeros( (h,w), np.float32 )
        mapv = np.zeros( (h,w), np.float32 )
        
        # 高さ方向に回す        
        for j in range(h):
            theta = np.pi * j / h
            cos_the = np.cos(theta)
            sin_the = np.sin(theta)
            
            # 正面
            for i in range(w//2):
                phi = 2.0 * np.pi * i / w
                cos_phi = np.cos(phi)
                sin_phi = np.sin(phi)
                
                _x = self._focus * (cos_phi*sin_the) / (sin_phi * sin_the + self._focus)
                _z = self._focus * cos_the / (sin_phi * sin_the + self._focus)

                u = -_z * self._radius + self._center_f[0]
                v =  _x * self._radius + self._center_f[1]

                mapu[j,(shift+i)%w] = u
                mapv[j,(shift+i)%w] = v
                
            # 後面
            for i in range(w//2, w):
                phi = 2.0 * np.pi * i / w
                cos_phi = np.cos(phi)
                sin_phi = np.sin(phi)
                _x = -self._focus * (cos_phi * sin_the) / (sin_phi*sin_the - self._focus)
                _z = -self._focus * cos_the / (sin_phi * sin_the - self._focus)
                
                _x2 = np.cos(self._dangle) * _x - np.sin(self._dangle) * _z
                _z2 = np.sin(self._dangle) * _x + np.cos(self._dangle) * _z

                u = _z2 * self._radius + self._center_b[0]
                v = _x2 * self._radius + self._center_b[1]

                mapu[j,(shift+i)%w] = u
                mapv[j,(shift+i)%w] = v


        # 注目ラインだけ取り出す
        mapu_focused = np.empty( shape=(0,w), dtype = 'float32' )
        mapv_focused = np.empty( shape=(0,w), dtype = 'float32' )
        for j in self._imgline:
            h1 = j*self._blobsize
            mapu_focused = np.append( mapu_focused, mapu[h1:h1+self._blobsize, :], axis=0)
            mapv_focused = np.append( mapv_focused, mapv[h1:h1+self._blobsize, :], axis=0)

        # 高速化のため整数マップに変換
        map1, map2 = cv2.convertMaps( mapu_focused, mapv_focused, cv2.CV_16SC2 )
                                    
        # 終了
        return map1, map2


        
        
if __name__ == '__main__':
    main();