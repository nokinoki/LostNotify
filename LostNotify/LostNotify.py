
import sys
import numpy as np
import cv2
import time
import pdb

global isPlay
global spendFrameCounter

# ダミーcallback
def nothing(x):
    pass

# 再生ウィンドウをクリックした時のcallback
def onMouse(event, x, y, flags, param):
    global isPlay
    global spendFrameCounter

    if event == cv2.EVENT_RBUTTONDOWN:
        isPlay = not isPlay
    if event == cv2.EVENT_MBUTTONDOWN:
        # フレームNoを記録
        print (spendFrameCounter)
        file = open("value-time.txt","a")
        file.write(str(spendFrameCounter) + " ")
        file.close()

def main():
    global spendFrameCounter
    global isPlay

    args = sys.argv    

    domainExpand = 100
    miniPerimeter = 160
    maxPerimeter = 240

    # 動画の読み込み
    selectedInputIndex = int(args[1])
    inputsLibrary = [
        ("a0",150,233),
        ("a1",120,183),
        ("b0",97,174),
        ("b1",130,223),
        ("c0",146,221),
        ("c1",171,278),
        ("d0",168,256),
        ("d1",151,288),
        ("d2",121,227)]
    fileName,startFrameNo,endFrameNo = inputsLibrary[selectedInputIndex]
    # ファイル名を記録
    file = open("value-time.txt","a")
    file.write(fileName + ": ")
    file.close()
    caputuredFrame = cv2.VideoCapture("..\\..\\" + fileName + ".mp4")
    # WindowUIの作成
    cv2.namedWindow('image')
    cv2.namedWindow('Param',cv2.WINDOW_NORMAL)
    cv2.createTrackbar('H','Param',0,255,nothing)
    cv2.setTrackbarPos('H','Param',0)
    cv2.createTrackbar('Range','Param',1,255,nothing)
    cv2.setTrackbarPos('Range','Param',12)
    cv2.createTrackbar('S','Param',0,255,nothing)
    cv2.setTrackbarPos('S','Param',54)
    cv2.createTrackbar('Range2','Param',1,255,nothing)
    cv2.setTrackbarPos('Range2','Param',39)
    cv2.createTrackbar('V','Param',0,255,nothing)
    cv2.setTrackbarPos('V','Param',255)
    cv2.createTrackbar('Range3','Param',1,255,nothing)
    cv2.setTrackbarPos('Range3','Param',53)
    cv2.createTrackbar('Med','Param',1,255,nothing)
    cv2.setTrackbarPos('Med','Param',16)
    cv2.createTrackbar('Op','Param',1,255,nothing)
    cv2.setTrackbarPos('Op','Param',16)
    cv2.setMouseCallback('image',onMouse)
    # Bufferの初期化
    generalBuffer = np.ones([1280,720], 'uint8') * 255
    skinBuffer = np.zeros([720,1280,3], 'uint8')
    multiSkinBuffer = np.zeros([720,1280,3], 'uint8')
    background = np.zeros([720,1280,3], 'uint8')
    # 再生スタート
    spendFrameCounter = 0
    isPlay = True

    while(True):
        # スライダーの読み取り
        h = cv2.getTrackbarPos('H','Param')
        rangeForH = cv2.getTrackbarPos('Range','Param')
        s = cv2.getTrackbarPos('S','Param')
        rangeForS = cv2.getTrackbarPos('Range2','Param')
        v = cv2.getTrackbarPos('V','Param')
        rangeForV = cv2.getTrackbarPos('Range3','Param')
        medianParameter = cv2.getTrackbarPos('Med','Param')
        opningParameter = cv2.getTrackbarPos('Op','Param')

        # フレームを取得
        if isPlay:
            _, originalFrame = caputuredFrame.read()
            spendFrameCounter = spendFrameCounter + 1
        if originalFrame is None:
            break


        frame = originalFrame
        if True: # False: 処理のバイパス（動画の再生だけする）

            ###############
            ## 手の軌跡を追う
            ###############
            frame = originalFrame
            # HSV変換
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hGray = hsv ** np.array([1,1,1],'uint8')
            hGray = hGray + np.array([0,0,0], 'uint8') 
            hGray = cv2.medianBlur(hGray, ksize=5)

            # HSVで閾値処理
            minH = np.array([max(h-rangeForH,0),max(s-rangeForS,0),max(v-rangeForV,0)])
            maxH = np.array([min(h+rangeForH,255),min(s+rangeForS,255),min(v+rangeForV,255)])
            mask = cv2.inRange(hGray, minH, maxH)
        
            if spendFrameCounter < 2:
                # 1フレーム目は背景を取得だけして飛ばす
                kernel = np.ones((5,5),np.uint8)
                background = cv2.dilate(mask,kernel,iterations = 10)
                background = cv2.bitwise_not(background)
                continue
            else:
                # 背景を消す
                mask = cv2.bitwise_and(mask,background)
        
            # オープニング
            kernel = np.ones((medianParameter//2+1,medianParameter//2+1),np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            # 手の奇跡を蓄積
            if spendFrameCounter > startFrameNo and spendFrameCounter < endFrameNo:
                skinBuffer = np.zeros([720,1280,3], 'uint8')
                skinBuffer = skinBuffer + np.array([spendFrameCounter//2,254,127],'uint8')
                skinBuffer = cv2.cvtColor(skinBuffer, cv2.COLOR_HSV2RGB)
                newBlue = cv2.bitwise_and(skinBuffer,skinBuffer, mask=mask)
                multiSkinBuffer = cv2.bitwise_or(newBlue,multiSkinBuffer)

            kernel = np.ones((5,5),np.uint8)
            maskLine = cv2.erode(multiSkinBuffer,kernel,iterations = 5)
            binLine = cv2.cvtColor(maskLine, cv2.COLOR_RGB2HSV).dot(np.array([0,0,1],'uint8'))
            edge = cv2.cvtColor(cv2.Canny(binLine, 50, 150), cv2.COLOR_GRAY2BGR)

            # ラベリング
            labelnum, labelimg, rectsContour, GoCs = cv2.connectedComponentsWithStats(binLine)
            necks = [False] * (labelnum + 1)
            parents = [None] * (labelnum + 1)
            for label in range(1,labelnum):
                gx,gy = GoCs[label]
                x,y,w,h,size = rectsContour[label]
                if size <= 100:
                    continue 
                if gy < 100:
                    necks[label] = True 
                else: 
                    necks[label] = False
            for label in range(1,labelnum):
                if necks[labelnum]:
                    continue
                gx,gy = GoCs[label]
                gocVector = np.array([gx,gy]);
                x,y,w,h,size = rectsContour[label]
                nearNeck = -1
                nearDist = 100000
                for neck in range(1,labelnum):
                    if not necks[neck]:
                        break
                    px,py = GoCs[neck]
                    dist = np.linalg.norm(gocVector - np.array([px,py]))
                    if dist < nearDist :
                        nearDist = dist
                        nearNeck = neck
                parents[label] = nearNeck
        
            ###############
            ## スマホを見つける
            ###############
            frame = originalFrame
            # 輪郭抽出
            gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
            delta = cv2.Canny(gray,50,110)
            _,countours,_ = cv2.findContours(delta,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            # 輪郭の単純化（凸に変形）
            for i in range(0,len(countours)):
                countour = countours[i]
                countour = cv2.convexHull(countour)
                countours[i] = countour
            # 描画用のBuffer
            articlesBuffer = np.zeros([720,1280,3], 'uint8')
            for i in range(0,len(countours)):
                perimeter = cv2.arcLength(countours[i],False)
                if perimeter < miniPerimeter or perimeter > maxPerimeter:# or cv2.contourArea(cons[i]) < ariaMin:
                    continue
                colorIndex = i%256
                hsv = np.array([[[colorIndex,255,255]]],'uint8')
                rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
                rgbTuple = (int(rgb[0,0,0]),int(rgb[0,0,1]),int(rgb[0,0,2]))
                articlesBuffer = cv2.drawContours(articlesBuffer,countours,i, rgbTuple)
                cx = countours[i][0,0,0]
                cy = countours[i][0,0,1]
                text =  "#" + str(i) + "-x:" + str(cx) + "-y:" + str(cy)
                articlesBuffer = cv2.putText(articlesBuffer,text,(cx,cy),cv2.FONT_HERSHEY_SIMPLEX, 0.3,rgbTuple)

            ###############
            ## 判定
            ###############
            havings = [-1] * (labelnum + 1)
            havingsInformation = ["-" for i in range(0,labelnum+1)]
            if spendFrameCounter > startFrameNo:
                for label in range(1,labelnum):
                    if necks[labelnum]:
                        continue
                    havings[label] = 0
                    havingsInformation[label] = ""
                    gx,gy = GoCs[label]
                    gocVector = np.array([gx,gy]);
                    x,y,w,h,size = rectsContour[label]
                    for i in range(0,len(countours)):
                        countour = countours[i]
                        perimeter = cv2.arcLength(countour,False)
                        if perimeter < miniPerimeter or perimeter > maxPerimeter:# or cv2.contourArea(cons[i]) < ariaMin:
                            continue
                        inIncluded = False
                        for j in range(0, len(countour)):
                            cx = countour[j,0,0]
                            cy = countour[j,0,1]
                            includedX = cx > x and x+w > cx
                            includedY = cy > y and y+h+domainExpand > cy
                            if includedX and includedY:
                                inIncluded = True
                                break
                        if inIncluded:
                            havings[label] = havings[label] + 1
                            havingsInformation[label] = havingsInformation[label] + "/#" + str(i) + "-x:" + str(cx) + "-y:" + str(cy) 
                        
            ###############
            ## 表示
            ###############
            frame = originalFrame
            # 見やすいようにグレースケール
            frame = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
            # 小物の表示を描画
            frame = cv2.bitwise_or(articlesBuffer,frame)
            # 手肌追跡を描画
            if spendFrameCounter > startFrameNo:
                # 手の軌跡をハイライト
                frame = frame - maskLine // 3
                frame = cv2.bitwise_or(edge,frame)
                for label in range(1,labelnum):
                    gx,gy = GoCs[label]
                    x,y,w,h,size = rectsContour[label]
                    if size <= 100:
                        continue 
                    #frame = cv2.circle(frame, (int(gx),int(gy)), 1, (0,255,0) if necks[label] else (0,0,255), -1)    
                    rectColor = (0,255,0) if necks[label] else ((255,0,0) if not havings[label] > 0 else (0,0,255))
                    frame = cv2.rectangle(frame, (x,y), (x+w,y+h+ (0 if necks[label] else domainExpand) ), rectColor,1 if not havings[label] else 2)
                    domainText =  "Domain(" + str(label) + ") having " + str(havings[label])
                    frame = cv2.putText(frame,domainText,(x,y + (h+10 if necks[label] else -10)),cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0,255,0) if necks[label] else (0,0,255))
                    # 所属を示す線を描画
                    if not necks[label]:
                        px,py = GoCs[parents[label]]
                        frame = cv2.line(frame,(int(gx),int(gy)),(int(px),int(py)),(255,0,0))
            frame = cv2.line(frame,(0,100),(1279,100),(255,31,127))
        
        frame = cv2.putText(frame,"VTime" if spendFrameCounter > startFrameNo and spendFrameCounter < endFrameNo else "---",(1110,700),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,0),3);
        cv2.imshow('image',frame)

        # qキーが押されたら途中終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if not caputuredFrame.isOpened():
            isPlay = False

    cv2.waitKey(1)
    caputuredFrame.release()
    cv2.destroyAllWindows()
    file = open("value-time.txt","a")
    file.write("\n")
    file.close()

if __name__ == "__main__":
    main()