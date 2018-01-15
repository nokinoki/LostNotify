
import sys
import numpy as np
import cv2
import time
from matplotlib import pyplot as plt
import pdb

global isPlay
global counter

def nothing(x):
    pass

def onMouse(event, x, y, flags, param):
    global isPlay
    global counter

    if event == cv2.EVENT_RBUTTONDOWN:
        #print(x,y)
        isPlay = not isPlay
    if event == cv2.EVENT_MBUTTONDOWN:
        print (counter)
        f = open("value-time.txt","a")
        f.write(str(counter) + " ")
        f.close()

def main():
    global counter
    global isPlay

    args = sys.argv    

    domainEx = 100
    perimeterLow = 160
    perimeterHeigh = 240
    ariaMin = 100

    # 動画の読み込み
    selected = int(args[1])
    videoLib = [
        ("a0",150,233),
        ("a1",120,183),
        ("b0",97,174),
        ("b1",130,223),
        ("c0",146,221),
        ("c1",171,278),
        ("d0",168,256),
        ("d1",151,288),
        ("d2",121,227)]

        # original
        #a0: 150 233 
        #a1: 120 183 
        #b0: 97 174 
        #b1: 130 223 
        #c0: 146 221 
        #c1: 171 278 
        #d0: 168 256 
        #d1: 151 288 
        #d2: 121 227 

    fname,startP,endP = videoLib[selected]
    f = open("value-time.txt","a")
    f.write(fname + ": ")
    f.close()
    cap = cv2.VideoCapture("..\\..\\" + fname + ".mp4")
    # 再生スタート
    isPlay = True

    # Create a black image, a window
    img = np.zeros((1280,720,3), np.uint8)
    cv2.namedWindow('image')
    cv2.namedWindow('Param',cv2.WINDOW_NORMAL)

    # create trackbars for color change
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

    #pdb.set_trace()

    white = np.ones([1280,720], 'uint8') * 255
    blue = np.zeros([720,1280,3], 'uint8')
    multiMask = np.zeros([720,1280,3], 'uint8')
    background = np.zeros([720,1280,3], 'uint8')

    # 動画終了まで繰り返し
    counter = 0
    while(True):

        #print(counter)

        # get current positions of four trackbars
        h = cv2.getTrackbarPos('H','Param')
        r = cv2.getTrackbarPos('Range','Param')
        s = cv2.getTrackbarPos('S','Param')
        r2 = cv2.getTrackbarPos('Range2','Param')
        v = cv2.getTrackbarPos('V','Param')
        r3 = cv2.getTrackbarPos('Range3','Param')
        m = cv2.getTrackbarPos('Med','Param')
        o = cv2.getTrackbarPos('Op','Param')

        # フレームを取得
        if isPlay:
            _, originalFrame = cap.read()
            counter = counter + 1
        if originalFrame is None:
            break

        ###############
        ## 手の軌跡を追う
        ###############

        frame = originalFrame
        
        if True: # False: バイパス

            # HSV変換
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            h00 = hsv ** np.array([1,1,1],'uint8')
            h00 = h00 + np.array([0,0,0], 'uint8') 
            h00 = cv2.medianBlur(h00, ksize=5)

            # HSVで閾値処理
            low = np.array([max(h-r,0),max(s-r2,0),max(v-r3,0)])
            heigh = np.array([min(h+r,255),min(s+r2,255),min(v+r3,255)])
            mask = cv2.inRange(h00, low, heigh)
        
            if counter < 2:
                kernel = np.ones((5,5),np.uint8)
                background = cv2.dilate(mask,kernel,iterations = 10)
                background = cv2.bitwise_not(background)
                continue
            else:
                mask = cv2.bitwise_and(mask,background)
        
            # オープニング
            kernel = np.ones((m//2+1,m//2+1),np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            # 手の奇跡を蓄積
            if counter > startP and counter < endP:
                blue = np.zeros([720,1280,3], 'uint8')
                blue = blue + np.array([counter//2,254,127],'uint8')
                blue = cv2.cvtColor(blue, cv2.COLOR_HSV2RGB)
                newBlue = cv2.bitwise_and(blue,blue, mask=mask)
                multiMask = cv2.bitwise_or(newBlue,multiMask)

            kernel = np.ones((5,5),np.uint8)
            maskLine = cv2.erode(multiMask,kernel,iterations = 5)
            binLine = cv2.cvtColor(maskLine, cv2.COLOR_RGB2HSV).dot(np.array([0,0,1],'uint8'))
            edge = cv2.cvtColor(cv2.Canny(binLine, 50, 150), cv2.COLOR_GRAY2BGR)

            # ラベリング
            labelnum, labelimg, contours, GoCs = cv2.connectedComponentsWithStats(binLine)
            necks = [False] * (labelnum + 1)
            parents = [None] * (labelnum + 1)
            for label in range(1,labelnum):
                gx,gy = GoCs[label]
                x,y,w,h,size = contours[label]
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
                gp = np.array([gx,gy]);
                x,y,w,h,size = contours[label]
                nearNeck = -1
                nearDist = 100000
                for neck in range(1,labelnum):
                    if not necks[neck]:
                        break
                    px,py = GoCs[neck]
                    dist = np.linalg.norm(gp - np.array([px,py]))
                    if dist < nearDist :
                        nearDist = dist
                        nearNeck = neck
                parents[label] = nearNeck


        
            ###############
            ## スマホを見つける
            ###############

            frame = originalFrame

            hsv = cv2.cvtColor(frame,cv2.COLOR_RGB2HSV)
            gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)

            delta = cv2.Canny(gray,50,110)
            cImg, cons, hir = cv2.findContours(delta,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            for i in range(0,len(cons)):
                con = cons[i]
                con = cv2.convexHull(con)
                #con = cv2.approxPolyDP(con,0.001*cv2.arcLength(con,True),True)
                cons[i] = con
        
            canv = np.zeros([720,1280,3], 'uint8')
            for i in range(0,len(cons)):
                perimeter = cv2.arcLength(cons[i],False)
                if perimeter < perimeterLow or perimeter > perimeterHeigh:# or cv2.contourArea(cons[i]) < ariaMin:
                    continue
                colIdx = i%256
                hsv = np.array([[[colIdx,255,255]]],'uint8')
                rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
                rgbtap = (int(rgb[0,0,0]),int(rgb[0,0,1]),int(rgb[0,0,2]))
                canv = cv2.drawContours(canv,cons,i, rgbtap)
                cx = cons[i][0,0,0]
                cy = cons[i][0,0,1]
                txt =  "#" + str(i) + "-x:" + str(cx) + "-y:" + str(cy)
                canv = cv2.putText(canv,txt,(cx,cy),cv2.FONT_HERSHEY_SIMPLEX, 0.3,rgbtap)



            ###############
            ## 判定
            ###############
            havings = [-1] * (labelnum + 1)
            havingsInfo = ["-" for i in range(0,labelnum+1)]
            if counter > startP:
                for label in range(1,labelnum):
                    if necks[labelnum]:
                        continue
                    havings[label] = 0
                    havingsInfo[label] = ""
                    gx,gy = GoCs[label]
                    gp = np.array([gx,gy]);
                    x,y,w,h,size = contours[label]
                    for i in range(0,len(cons)):
                        con = cons[i]
                        perimeter = cv2.arcLength(con,False)
                        if perimeter < perimeterLow or perimeter > perimeterHeigh:# or cv2.contourArea(cons[i]) < ariaMin:
                            continue
                        incl = False
                        for j in range(0, len(con)):
                            cx = con[j,0,0]
                            cy = con[j,0,1]
                            xincl = cx > x and x+w > cx
                            yincl = cy > y and y+h+domainEx > cy
                            if xincl and yincl:
                                incl = True
                                break
                        if incl:
                            havings[label] = havings[label] + 1
                            havingsInfo[label] = havingsInfo[label] + "/#" + str(i) + "-x:" + str(cx) + "-y:" + str(cy) 
                #print(havings)
                        
                        
                
            ###############
            ## 表示
            ###############
        
            frame = originalFrame

            # 見やすいようにグレースケール
            frame = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

            frame = cv2.bitwise_or(canv,frame)

            if counter > startP:
                # 手の軌跡をハイライト
                frame = frame - maskLine // 3
                frame = cv2.bitwise_or(edge,frame)
                for label in range(1,labelnum):
                    gx,gy = GoCs[label]
                    x,y,w,h,size = contours[label]
                    if size <= 100:
                        continue 
                    #frame = cv2.circle(frame, (int(gx),int(gy)), 1, (0,255,0) if necks[label] else (0,0,255), -1)    
                    recCol = (0,255,0) if necks[label] else ((255,0,0) if not havings[label] > 0 else (0,0,255))
                    frame = cv2.rectangle(frame, (x,y), (x+w,y+h+ (0 if necks[label] else domainEx) ), recCol,1 if not havings[label] else 2)
                    domainTxt =  "Domain(" + str(label) + ") having " + str(havings[label])
                    frame = cv2.putText(frame,domainTxt,(x,y + (h+10 if necks[label] else -10)),cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0,255,0) if necks[label] else (0,0,255))

                    if not necks[label]:
                        px,py = GoCs[parents[label]]
                        frame = cv2.line(frame,(int(gx),int(gy)),(int(px),int(py)),(255,0,0))
            frame = cv2.line(frame,(0,100),(1279,100),(255,31,127))
        
        frame = cv2.putText(frame,"VTime" if counter > startP and counter < endP else "---",(1110,700),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,0),3);
        cv2.imshow('image',frame)

        # qキーが押されたら途中終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if not cap.isOpened():
            isPlay = False

    cv2.waitKey(1)
    cap.release()
    cv2.destroyAllWindows()
    f = open("value-time.txt","a")
    f.write("\n")
    f.close()

if __name__ == "__main__":
    main()