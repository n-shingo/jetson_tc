# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 17:40:31 2017
@author: shingo
ThetaS画像から人を検出するクラス
"""

import cv2
import numpy as np

if( __name__ == '__main__' ):
    from thetas_focused_converter import ThetaSFocusedConverter
else:
    from tool.thetas_focused_converter import ThetaSFocusedConverter
    
from chainer import cuda, Variable

class PeopleSearcher:
    def __init__(self, cnn, gpu=0, threshold=0.9, saveimgdir=None):
        self.CNN = cnn
        self.CNN.train = False
        self.CNN.predictor.train = False

        self.gpu = gpu
        self.XP = cuda.cupy if gpu >= 0 else np
        self.Threshold = threshold
        #self.conv = ThetaSFocusedConverter(blobsize=32, hcnt=16, wcnt=32, imgline=[8,9,10])
        self.conv = ThetaSFocusedConverter(blobsize=16, hcnt=32, wcnt=64, imgline=[15,16,17,18,19,20])
        self.save_img_cnt = 0
        
        self.counter = 0;  # 現在何枚画像が入力されているか数えるカウンタ
        self.max_cnt = 3;  # 最大見る枚数
        self.psn_pos = list([0,])*self.max_cnt

        self.saveimg_dir = saveimgdir #画像保存ディレクトリ
        if self.saveimg_dir is not None and self.saveimg_dir[-1] != '/' :
            self.saveimg_dir += '/'
        
    
    # 画像から人を検出    
    def find_people( self, img, show_imgs=False, show_details=False, save_img=False ):
        
        # コンバータで正距円筒図に変換
        img = self.conv.convert(img);
        img_origin = img.copy()
        
        # 必要な変数
        h_org, w_org = img.shape[:2]
        assert h_org==96 and w_org==1024, "converted theta image must be 96 x 1024"
    
        # 後ろ90度カット(前方 270 度見る)画像
        img = img[:, 1024//8:1024*7//8,:]

        # 画像の保存
        if save_img and img is not None:
            filename = self.saveimg_dir + "img{0:06d}.png".format(self.save_img_cnt)
            cv2.imwrite( filename, img )
            self.save_img_cnt+=1

    
        # 遠くを検索するための画像作成
        img_far = img[0:64,:,:]
        h_far, w_far = img_far.shape[:2]
    
        # 近くを検索するための画像作成
        img_near = cv2.resize(img, (512, 64))
        h_near, w_near = img_near.shape[:2]
        
        
        # CNN入力サイズ
        insize = 64
        
        #横のブロック数
        cnt_far = 2*( w_far // insize ) - 1
        cnt_near = 2*( w_near // insize ) - 1
        
        # 原画像分割
        blob_imgs = []
        
        # 遠く用の画像取得
        for i in range( cnt_far ):
            y1 = 0; x1 = i*insize//2;
            blob_imgs.append( img_far[y1:insize, x1:x1+insize,:] )
                
        # 近く用の画像取得
        for i in range( cnt_near ):
            y1 = 0; x1 = i*insize//2;
            blob_imgs.append( img_near[y1:insize, x1:x1+insize,:] )
    
    
        # --------------------居る居ないの判定 ---------------------------#      
    
        # cnn入力データ作成
        in_imgs = []
        for blob in blob_imgs:
            in_imgs.append( blob.astype(np.float32).transpose(2,0,1))
        in_imgs = self.XP.array(in_imgs)
        x = Variable( in_imgs )
                
        # cnn 前向き計算
        y = self.CNN.predictor(x)
        
        # 居る・居ない判定
        # blf[j,i] > 0.5 で判定できるので不要なコードとなったが、念のため取っておく
        # res = y.data.argmax(axis=1).reshape(blob_h, blob_w)
        
        # 信頼度を計算
        blf = self.XP.exp( y.data )
        blf = blf[:,1] / self.XP.sum( blf, axis=1 )
        blf_far = blf[:cnt_far]
        blf_near = blf[cnt_far:cnt_far+cnt_near]
    
        # 結果チェック    
        ret_img_far = img_far.copy()//2  # 全体を暗くしてコピー
        for i in range( cnt_far ):
            if blf_far[i] > self.Threshold:
                # 閾値以上であれば元の明るさに戻す
                lft = i*insize//2;
                ret_img_far[:, lft:lft+insize,:] = img_far[:, lft:lft+insize,:]
    
            if( show_details):
                self.stamp_text( '{:.1f}'.format(float(blf_far[i])), blob_imgs[i] )
                cv2.imshow("far"+str(i), blob_imgs[i])
                cv2.moveWindow( "far"+str(i), i*80, 0 )
    
    
        # 一箇所もなければ関数終了
        #if( len( blf[blf>=THRESHOLD]) == 0 ):
        #   return ret_img_far
    
        ret_img_near = img_near.copy()//2  # 全体を暗くしてコピー
        for i in range( cnt_near ):
            if blf_near[i] > self.Threshold:
                # 閾値以上であれば元の明るさに戻す
                lft = i*insize//2;
                ret_img_near[:, lft:lft+insize,:] = img_near[:, lft:lft+insize,:]
            
            if( show_details ):
                self.stamp_text( '{:.1f}'.format(float(blf_near[i])), blob_imgs[cnt_far+i] )
                cv2.imshow("near"+str(i), blob_imgs[cnt_far+i])
                cv2.moveWindow( "near"+str(i), i*80, 100 )
                
                
        # 位置判定
        pos_rad = None
        if self.gpu>=0: # cupy -> numpy
            blf_far = cuda.to_cpu( blf_far ).tolist()
            blf_near = cuda.to_cpu( blf_near).tolist()
            
        if( max(blf) > self.Threshold ):
            max_far = max(blf_far)
            max_near = max(blf_near)
            
            if max_far > max_near :
                index = blf_far.index( max_far )
                pos_rad = (index-11) * 135 / 12.0 * np.pi / 180.0
                pos_pix = 32 + index*32
                cv2.circle( ret_img_far, (pos_pix, 32), 20, (0,0,255) )
                
            else:
                index = blf_near.index( max_near )
                pos_rad = (index-7) * 135 / 8.0 * np.pi / 180.0
                pos_pix = 32 + index*32
                cv2.circle( ret_img_near, (pos_pix, 32), 20, (0,0,255) )
                
        # 発見した位置を記憶する
        self.psn_pos[self.counter] = pos_rad


        # 処理画像の表示
        if show_imgs:
            cv2.imshow( "far image", ret_img_far )
            cv2.moveWindow( "far image", 0, 200 )
            cv2.imshow( "near image", ret_img_near)
            cv2.moveWindow( "near image", 0, 300 )
        
        if show_imgs or show_details:
            cv2.waitKey(1)
            
        # img に円を描く
        if pos_rad is not None:
            pos_pix = np.round(pos_rad * 180.0 / np.pi / 360.0 * img_origin.shape[1] + img_origin.shape[1]//2)
            cv2.circle( img_origin, (int(pos_pix), img_origin.shape[0]//2), 40, (0,0,255))
            pos_rad = -pos_rad
            
            
        # count up image counter
        self.counter += 1
        
        # check counter
        if self.max_cnt == self.counter :
            self.counter = 0

        # return values
        return img_origin, pos_rad
        
        """
        # -------------------- 位置の判定 ---------------------------#
        
        # 位置の判定
        max_idx = max_j*blob_w+max_i
        x = Variable(in_imgs[max_idx:max_idx+1])
        y = area_cnn(x)    
        if GPU >= 0:
            y = cuda.to_cpu( y.data )[0]
        else:
            y = y.data[0]
    
    
        # 四角描画
        top = max_j*insize//2; lft = max_i*insize//2;
        blob_img = ret_img[top:top+insize, lft:lft+insize,:]
        x1 = np.round(y[0])
        y1 = np.round(y[1])
        x2 = np.round(y[2])
        y2 = np.round(y[3])
        cv2.rectangle( blob_img, (x1,y1), (x2,y2), (0,0,255) )
        
        return ret_img
    """

    # 画像から人を検出    
    def find_people_multiimg( self, img, show_imgs=False, show_details=False, save_img=False ):
        
        # コンバータで正距円筒図に変換
        img = self.conv.convert(img);
        img_origin = img.copy()
        
        # 必要な変数
        h_org, w_org = img.shape[:2]
        assert h_org==96 and w_org==1024, "converted theta image must be 96 x 1024"
    
        # 後ろ90度カット(前方 270 度見る)画像
        img = img[:, 1024//8:1024*7//8,:]

        # 画像の保存
        if save_img and img is not None:
            filename = self.saveimg_dir + "img{0:06d}.png".format(self.save_img_cnt)
            cv2.imwrite( filename, img )
            self.save_img_cnt+=1

    
        # 遠くを検索するための画像作成
        img_far = img[0:64,:,:]
        h_far, w_far = img_far.shape[:2]
    
        # 近くを検索するための画像作成
        img_near = cv2.resize(img, (512, 64))
        h_near, w_near = img_near.shape[:2]
        
        
        # CNN入力サイズ
        insize = 64
        
        #横のブロック数
        cnt_far = 2*( w_far // insize ) - 1
        cnt_near = 2*( w_near // insize ) - 1
        
        # 原画像分割
        blob_imgs = []
        
        # 遠く用の画像取得
        for i in range( cnt_far ):
            y1 = 0; x1 = i*insize//2;
            blob_imgs.append( img_far[y1:insize, x1:x1+insize,:] )
                
        # 近く用の画像取得
        for i in range( cnt_near ):
            y1 = 0; x1 = i*insize//2;
            blob_imgs.append( img_near[y1:insize, x1:x1+insize,:] )
    
    
        # --------------------居る居ないの判定 ---------------------------#      
    
        # cnn入力データ作成
        in_imgs = []
        for blob in blob_imgs:
            in_imgs.append( blob.astype(np.float32).transpose(2,0,1))
        in_imgs = self.XP.array(in_imgs)
        x = Variable( in_imgs )
                
        # cnn 前向き計算
        y = self.CNN.predictor(x)
        
        # 居る・居ない判定
        # blf[j,i] > 0.5 で判定できるので不要なコードとなったが、念のため取っておく
        # res = y.data.argmax(axis=1).reshape(blob_h, blob_w)
        
        # 信頼度を計算
        blf = self.XP.exp( y.data )
        blf = blf[:,1] / self.XP.sum( blf, axis=1 )
        blf_far = blf[:cnt_far]
        blf_near = blf[cnt_far:cnt_far+cnt_near]
    
        # 結果チェック    
        ret_img_far = img_far.copy()//2  # 全体を暗くしてコピー
        for i in range( cnt_far ):
            if blf_far[i] > self.Threshold:
                # 閾値以上であれば元の明るさに戻す
                lft = i*insize//2;
                ret_img_far[:, lft:lft+insize,:] = img_far[:, lft:lft+insize,:]
    
            if( show_details):
                self.stamp_text( '{:.1f}'.format(float(blf_far[i])), blob_imgs[i] )
                cv2.imshow("far"+str(i), blob_imgs[i])
                cv2.moveWindow( "far"+str(i), i*80, 0 )
    
    
        # 一箇所もなければ関数終了
        #if( len( blf[blf>=THRESHOLD]) == 0 ):
        #   return ret_img_far
    
        ret_img_near = img_near.copy()//2  # 全体を暗くしてコピー
        for i in range( cnt_near ):
            if blf_near[i] > self.Threshold:
                # 閾値以上であれば元の明るさに戻す
                lft = i*insize//2;
                ret_img_near[:, lft:lft+insize,:] = img_near[:, lft:lft+insize,:]
            
            if( show_details ):
                self.stamp_text( '{:.1f}'.format(float(blf_near[i])), blob_imgs[cnt_far+i] )
                cv2.imshow("near"+str(i), blob_imgs[cnt_far+i])
                cv2.moveWindow( "near"+str(i), i*80, 100 )
                
                
        # 位置判定
        pos_rad = None
        if self.gpu>=0: # cupy -> numpy
            blf_far = cuda.to_cpu( blf_far ).tolist()
            blf_near = cuda.to_cpu( blf_near).tolist()
            
        if( max(blf) > self.Threshold ):
            max_far = max(blf_far)
            max_near = max(blf_near)
            
            if max_far > max_near :
                index = blf_far.index( max_far )
                pos_rad = -(index-11) * 135 / 12.0 * np.pi / 180.0
                pos_pix = 32 + index*32
                cv2.circle( ret_img_far, (pos_pix, 32), 20, (0,0,255) )
                
            else:
                index = blf_near.index( max_near )
                pos_rad = -(index-7) * 135 / 8.0 * np.pi / 180.0
                pos_pix = 32 + index*32
                cv2.circle( ret_img_near, (pos_pix, 32), 20, (0,0,255) )
                

        # img に赤円を描く
        img_origin2 = img_origin.copy()
        if pos_rad is not None:
            pos_pix = np.round(-pos_rad * 180.0 / np.pi / 360.0 * img_origin.shape[1] + img_origin.shape[1]//2)
            cv2.circle( img_origin2, (int(pos_pix), img_origin.shape[0]//2), 40, (0,0,255))

        # 処理画像の表示
        if show_imgs:
            cv2.imshow( "far image", ret_img_far )
            cv2.moveWindow( "far image", 0, 200 )
            cv2.imshow( "near image", ret_img_near)
            cv2.moveWindow( "near image", 0, 300 )
            cv2.imshow( "result image", img_origin2 )
            cv2.moveWindow( "result image", 0, 400 )
        
        if show_imgs or show_details:
            cv2.waitKey(1)
            
            
            
        # 発見した位置を記憶する
        if pos_rad is not None:
            self.psn_pos[self.counter] = pos_rad
        else:
            self.psn_pos[self.counter] = None

        # count up image counter
        self.counter += 1
        
        # 計測に十分なデータが集まったら複数から位置を割り出す
        total_pos_rad = None
        if self.counter == self.max_cnt:
            pos_s = [e for e in self.psn_pos if e is not None]  # Noneでない要素だけ取り出す
            # 半分以上が有効な数値の場合は処理をする
            if( len(pos_s)/float(self.max_cnt) > 0.5 ):
                dp = []
                for i in range(len(pos_s)-1):
                    dp.append(abs(pos_s[i+1]-pos_s[i]))
                inside = True
                for d in dp:
                    if d > np.pi * 17.0 / 180.0:  # 17.0degree
                        inside = False
                        break
                if inside:
                    total_pos_rad = sum(pos_s)/len(pos_s)

        # img に緑円を描く
        if total_pos_rad is not None:
            pos_pix = np.round(-total_pos_rad * 180.0 / np.pi / 360.0 * img_origin.shape[1] + img_origin.shape[1]//2)
            cv2.circle( img_origin, (int(pos_pix), img_origin.shape[0]//2), 40, (0,255,0), thickness=2)

        # return values
        if self.max_cnt == self.counter :
            self.counter = 0
            return True, img_origin, total_pos_rad
        else:
            return False, None, None
        
        """
        # -------------------- 位置の判定 ---------------------------#
        
        # 位置の判定
        max_idx = max_j*blob_w+max_i
        x = Variable(in_imgs[max_idx:max_idx+1])
        y = area_cnn(x)    
        if GPU >= 0:
            y = cuda.to_cpu( y.data )[0]
        else:
            y = y.data[0]
    
    
        # 四角描画
        top = max_j*insize//2; lft = max_i*insize//2;
        blob_img = ret_img[top:top+insize, lft:lft+insize,:]
        x1 = np.round(y[0])
        y1 = np.round(y[1])
        x2 = np.round(y[2])
        y2 = np.round(y[3])
        cv2.rectangle( blob_img, (x1,y1), (x2,y2), (0,0,255) )
        
        return ret_img
    """

    # 画像に文字列をスタンプ(刻印する)
    def stamp_text( self, text, image ):
    
        loca = (0,20)    
        font = cv2.FONT_HERSHEY_SIMPLEX
        size = 0.6
        color = (0,0,255)
        
        cv2.putText(image, text, loca, font, size, color)
        
