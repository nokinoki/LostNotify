# coding:utf-8
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import time
import numpy as np

# 背景を学習するフレーム数
BUFFER_LEN = 10
# 撮影サイズ
WIDTH = 32
HEIGHT = 32
SCALE = 8
SHOW_SIZE = (SCALE*WIDTH, SCALE*HEIGHT)
# 閾値
THRESHOLD = 31

# 入力そのままのWindow
cv2.cv.NamedWindow("Input", 1)
# 背景と認識した画像のWindow
cv2.cv.NamedWindow("Background", 1)
# 入力-背景のWindow
cv2.cv.NamedWindow("Diff between Input, Background", 1)
# 入力-背景を二値化したWindow
cv2.cv.NamedWindow("Threshold", 1)

# cameraオブジェクトのセクション
with PiCamera() as camera:

    # モジュールの初期化
    camera.resolution = (WIDTH, HEIGHT)
    rawCapture = PiRGBArray(camera, size = (WIDTH, HEIGHT))
    time.sleep(0.1)

    loopCounter = 0
    # 学習した背景オブジェクト

    # メインループ
    for frame in camera.capture_continuous(rawCapture, format="rgb", use_video_port = True):

        # 画像のマトリクスを取得
        rawImage = frame.array
        rawImage = cv2.flip(rawImage, 0)
        # 平滑化
        #rawImage = cv2.medianBlur(rawImage, 3)

        # 表示
        cv2.imshow("Input", cv2.resize(rawImage, SHOW_SIZE, interpolation=cv2.INTER_NEAREST))

        # グレースケール加工
        grayImage = cv2.cvtColor(rawImage, cv2.COLOR_RGB2GRAY)

        # 背景学習
        if loopCounter == 10:
            background = grayImage
            # 表示
            cv2.imshow("Background", cv2.resize(background, SHOW_SIZE, interpolation=cv2.INTER_NEAREST))
        if loopCounter >= 10:
            # 背景との差分
            diff = cv2.absdiff(grayImage, background)
            #diff = cv2.bitwise_and(rawImage, rawImage, mask=backgroundMask)
            # 表示
            cv2.imshow("Diff between Input, Background", cv2.resize(diff, SHOW_SIZE, interpolation=cv2.INTER_NEAREST))

            # 二値化
            _, threshold = cv2.threshold(diff, THRESHOLD, 255, cv2.THRESH_BINARY)
            cv2.imshow("Threshold", cv2.resize(threshold, SHOW_SIZE, interpolation=cv2.INTER_NEAREST))

            # ヒストグラム出力
            histgram = len(np.where(threshold>THRESHOLD)[0])
            if histgram < 24 and histgram > 3 :
                print "Thing?"
            if histgram > 25:
                print "Human?" 

        # 次のフレームまで待機
        key = cv2.waitKey(33) & 0xFF
        # フレームの破棄
        rawCapture.truncate(0)

        # メインループ脱出
        if key == ord("q"):
            break

        loopCounter = loopCounter + 1

# Windowの破棄
cv2.cv.DestroyAllWindows()
