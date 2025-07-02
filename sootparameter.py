#-*- coding : utf-8-*-

import cv2
import numpy as np  
import copy
#import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

showImage = 0
resizeRate = 0.6
barType = "bd"


def dist_Euc(point1,point2):
    d = np.float32(point1[0])-np.float32(point2[0])
    d = d**2
    p = np.float32(point1[1])-np.float32(point2[1])
    p = p**2
    d = d+p
    d = np.sqrt(d)
    return d

def takeX(elem):
    return elem[0]

def lineIntPol(y,P0,P1):            
    x0 = P0[0]
    y0 = P0[1]
    x1 = P1[0]
    y1 = P1[1]
    return x0+(y-y0)*(x1-x0)/(y1-y0)

def calAvgRad(circList):
    sumX = 0
    sumY = 0
    sumM = 0
    avgR = 0
    for circ in circList:
        avgR += circ[2]
    avgR = avgR/len(circList)
    for circ in circList:
        x = circ[0]
        y = circ[1]
        m = (circ[2]/avgR)**3
        sumX += m*x
        sumY += m*y
        sumM += m
    avgX = sumX/sumM
    avgY = sumY/sumM
    return [avgX,avgY]
    
def drawBorder(event, x, y, flags, param):
    #if event==cv2.EVENT_MOUSEMOVE and flags==cv2.EVENT_FLAG_LBUTTON:
    global partIndex
    continueFlag = 0
    if flags==cv2.EVENT_FLAG_LBUTTON:
        if len(routeList)==0:
            
            routeList.append([])
            partIndex = 0            
        else:
            for idx in range(len(routeList)):
                if dist_Euc(routeList[idx][-1],[int(x/resizeRate),int(y/resizeRate)])<=20/resizeRate:
                    partIndex = idx
                    continueFlag = 1
            if continueFlag == 0:                
                routeList.append([])
                partIndex = len(routeList)-1  
        routeList[partIndex].append([int(x/resizeRate),int(y/resizeRate)])
        
    if event == cv2.EVENT_LBUTTONUP:
        #print("LUP")
        i = 0
        while i <len(routeList):
            route1 = routeList[i]
            if len(route1)==1:
                del routeList[i]
            else:
                i = i+1
        i = 0
        while i <len(routeList):
            delFlag = 0
            route1 = routeList[i]
            for route0 in routeList:
                if routeList.index(route0)!=routeList.index(route1) and dist_Euc(route0[-1],route1[0])<=30/resizeRate:
                    routeList[routeList.index(route0)].extend(route1)
                    del routeList[i]
                    delFlag = 1
                    break
            if not delFlag:
                i = i+1
            
        
def showCord(event, x, y, flags, param):
    global mouseMaskd
    if event==cv2.EVENT_MOUSEMOVE:
        mouseMaskd[0:20,0:100]=0
        cv2.putText(mouseMaskd, str(x)+", "+str(y), (10,10), cv2.FONT_HERSHEY_COMPLEX, 0.5,255)

def getBarLength(img_ori,barType):
    length = 0
    if barType == 'fx':
        barArea = img_ori[int(0.97*img_ori.shape[0]):img_ori.shape[0],int(0.52*img_ori.shape[1]):img_ori.shape[1]]
        plt.imshow(barArea,"gray")
        plt.show()
        #print(barArea)
        thre,barArea_div = cv2.threshold(barArea,127,255, cv2.THRESH_BINARY) 
        for i in range(1,barArea_div.shape[1]-1):
            if np.sum(barArea_div[:,i])<=(barArea.shape[0]-6)*255:
                if np.sum(barArea_div[:,i-1])==np.sum(barArea_div[:,i])+7*255 and np.sum(barArea_div[:,i+1])==np.sum(barArea_div[:,i])+4*255:
                    length = i
                if np.sum(barArea_div[:,i+1])==np.sum(barArea_div[:,i])+7*255 and np.sum(barArea_div[:,i-1])==np.sum(barArea_div[:,i])+4*255:
                    length = i-length
    elif barType == 'bd':
        barArea = img_ori[int(0.85*img_ori.shape[0]):int(0.95*img_ori.shape[0]),int(0.0*img_ori.shape[1]):int(0.4*img_ori.shape[1])]
        thre,barArea_div = cv2.threshold(barArea,20,255, cv2.THRESH_BINARY)         
        plt.imshow(barArea_div,"gray")
        plt.show()
        #midx = np.min(np.sum(barArea_div,axis = 1))
        midx = np.argmin(np.sum(barArea_div,axis = 1), axis=0)
        print(midx)
        barList = []
        for p in range(1,barArea.shape[1]):
            if barArea_div[midx,p]==0 and barArea_div[midx,p-1]==255:
                barList.append(0)
            if barArea_div[midx,p]==0:
                barList[-1] +=1
        length = max(barList)
    return length

def dist_2Line(ptX,ptL1,ptL2):
    x = ptX[0]
    y = ptX[1]
    x1 = ptL1[0]
    y1 = ptL1[1]
    x2 = ptL2[0]
    y2 = ptL2[1]
    A = y2-y1
    B = x1-x2
    C = y1*x2-y2*x1
    return (A*x+B*y+C)/np.sqrt(A**2+B**2)

imgList = []
mskList = []
for root, dirs, files in os.walk(sys.path[0]): 
    for f in files:
        if f[-4:] == ".tif" or f[-4:] == ".TIF":
            imgList.append(f)
        elif len(f)>8 and (f[-8:] == "_Msk.png" or f[-8:] == "_Msk..TIF"):
            mskList.append(f)

print(imgList)   
for imgPath in imgList:
    imgScale = imgPath[0:-4].split("-")
    imgName = imgScale[0]
    if len(imgScale)==2 and (imgScale[1].isdigit()):
        imgScale = int(imgScale[1])
    elif len(imgScale)==3:
        if(imgScale[2].isdigit()):
            imgScale = int(imgScale[2])
        elif imgScale[2].isalpha():
            barType = imgScale[2]
        if imgScale[1].isalpha():
            barType = imgScale[1]
        elif(imgScale[1].isdigit()):
            imgScale = int(imgScale[1])
    else:
        imgScale = 0
        print("Unable to get Scale from File name, Please Check!")
        continue
        
    if barType not in ["bd","fx"]:
        print("The Type of Scale Bar is not defined or mis-defined, Please Check!")
        continue
        
    csvPath = imgPath[0:-3]+"csv"
    dtype = [("x",int),("y",int),("r",int),("ri",float)]
    print("Result will be saved into file: ",csvPath)

    print("loading Image Named ", imgPath)
    img_ori = cv2.imread(imgPath, 0)
    img_blur = cv2.GaussianBlur(img_ori,(3,3),0)# canny(): 
    canny_blur = cv2.Canny(img_blur, 50, 300)
    canny = cv2.Canny(img_ori,50, 350)
    pix2nm = imgScale/getBarLength(img_ori,barType)
    print("the resolution of the photograph is ",img_ori.shape)
    print("scaleBar: ",pix2nm)

    if showImage == True:
        cv2.imshow("img_ori",cv2.resize(img_ori,(0,0),fx = 0.5,fy = 0.5))
        cv2.imshow('Canny_AfterBlur', cv2.resize(canny_blur,(0,0),fx = 0.5,fy = 0.5))
        cv2.waitKey(0)

    print("Detecting Particles in Image...")
    cimg = []
    for i in range(0,3):
        cimg.append(cv2.cvtColor(img_ori,cv2.COLOR_GRAY2BGR))

    circList=[]
    riList = []
    circList.append(np.uint16(np.around(cv2.HoughCircles(img_blur,cv2.HOUGH_GRADIENT,1,7,param1=100,param2=5,minRadius=4,maxRadius=8)))[0])
    circList.append(np.uint16(np.around(cv2.HoughCircles(img_blur,cv2.HOUGH_GRADIENT,1,11,param1=100,param2=6,minRadius=8,maxRadius=12))[0]))

    j = 0
    while j<len(circList[0]):
        lc = circList[0][j]
        covFlag = 0
        for bc in circList[1]:
            if dist_Euc(lc[0:2],bc[0:2])<bc[2]:
                covFlag = 1
                #print("Circ at ",lc," covered by ",bc)
                break
        if covFlag == 1:
            circList[0] = np.delete(circList[0],j,axis = 0)
        else:
            j = j+1
            
    circListR = np.r_[circList[0],circList[1]]
    print("Detected ",len(circListR)," circles!");

    if showImage == True:
        p = 0
        for circ in circList:
            for i in circ:
                cv2.circle(cimg[p],(i[0],i[1]),i[2],(0,255,0),1)
                cv2.circle(cimg[p],(i[0],i[1]),1,(0,0,255),1)
                cv2.circle(cimg[2],(i[0],i[1]),i[2],(0,255,0),1)
                cv2.circle(cimg[2],(i[0],i[1]),1,(0,0,255),1) 
            p = p+1

        cv2.imshow('small circles',cimg[0])
        cv2.imshow('large circles',cimg[1])
        cv2.imshow('all circles',cimg[2])
        cv2.waitKey(0)

    print("Segementing Image...")
    thre,img_div = cv2.threshold(img_ori,0,255, cv2.THRESH_OTSU) 
    thre,img_div = cv2.threshold(img_blur,thre+42,255, cv2.THRESH_BINARY_INV) 
    imgScale = img_div.shape
    #th3 = cv2.adaptiveThreshold(img_blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,25,0)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))  # 
    img_divopn = cv2.morphologyEx(img_div, cv2.MORPH_OPEN, kernel)  # open
    img_divcls = cv2.morphologyEx(img_divopn, cv2.MORPH_CLOSE, kernel)  # close
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(6,6))
    img_divdil = cv2.dilate(img_divcls, kernel)

    if showImage == True:
        cv2.imshow('OTSU Devide',cv2.resize(img_div,(0,0),fx = 0.5,fy = 0.5))
        cv2.imshow('dilated Div',cv2.resize(img_divdil,(0,0),fx = 0.5,fy = 0.5))
        cv2.waitKey(0)
    
    print("Searching for Stored Routes...")
    if imgName+"_Msk.png" in mskList:
        mouseMaskd = cv2.imread(imgName+"_Msk.png",0)
    
    else:
        print("Routes NotFound, Please Draw the border of the patricle...")
        cv2.namedWindow('image')
        routeList = []
        partIndex = 0
        key = 0
        # Define the callback function of the mouse
        cv2.setMouseCallback('image', drawBorder)
        while(key != 27):
            key = cv2.waitKey(20)
            if key<=256 and key>=32 and chr(key)=='z' :
                print('z pressed')
                if partIndex>0:
                    del routeList[partIndex]
                    partIndex = partIndex-1
                else:
                    routeList.clear()
                
            img_blur1 = img_blur.copy()
            if(len(routeList)>=1):
                for route in routeList:
                    if(len(route)>=2):
                        for idx in range(len(route)-1):
                            cv2.line(img_blur1,(route[idx][0],route[idx][1]),(route[idx+1][0],route[idx+1][1]),(33, 33, 133),1);
                            cv2.circle(img_blur1,(route[idx][0],route[idx][1]),1,(0,0,255),-1)
                        cv2.circle(img_blur1,(route[-1][0],route[-1][1]),1,(0,0,255),-1)
            
            cv2.imshow('image',cv2.resize(img_blur1,(0,0),fx =resizeRate,fy =resizeRate))
            
        cv2.destroyAllWindows()
        
            
        print("Filling drawed Outlines...")
        routes = []
        for route in routeList:
            i = 0
            while i<len(route)-1:
                if route[i]==route[i+1]:
                    #print("deleted point:",route[i+1])
                    del route[i+1]
                else:
                    i = i+1
            routes.append(route)
        routeList = copy.deepcopy(routes)

        mouseMaskd = np.zeros(np.shape(img_ori), np.uint8)
        # mouseMaskd = img_ori.copy()
        if(len(routeList)>=1):
            for route in routeList:
                if(len(route)>=2):
                    for idx in range(len(route)-1):
                        cv2.line(mouseMaskd,(route[idx][0],route[idx][1]),(route[idx+1][0],route[idx+1][1]),0,1);
                        cv2.circle(mouseMaskd,(route[idx][0],route[idx][1]),1,0,-1)
                        mouseMaskd[route[idx][1],route[idx][0]] = 255
                    cv2.circle(mouseMaskd,(route[-1][0],route[-1][1]),1,0,-1)

        linePaintedFlag = np.zeros(np.shape(img_ori)[0])
        for curY in range(0,np.shape(img_ori)[0]): 
            pointInCol = []
            for route in routeList:
                if (route[0][1]>curY and curY>route[-1][1]) or (route[0][1]<curY and curY<route[-1][1]):  
                    newX = lineIntPol(curY,route[0],route[-1])
                    pointInCol.append(newX)   
                for point in route:
                    if point[1]==curY:
                        #print(curY,point)
                        pointInCol.append(point[0])
                    elif route.index(point)>0 and ((point[1]>curY and curY>route[route.index(point)-1][1])or(point[1]<curY and curY<route[route.index(point)-1][1])) :
                        newX = lineIntPol(curY,point,route[route.index(point)-1])
                        pointInCol.append(newX)   
            
            if len(pointInCol)>0:
                pointInCol.sort()
                #print("Found ",len(pointInCol)," Points in Row ",curY,":",pointInCol)
            
                
                actualPoint = len(pointInCol)
                actualPinC = copy.deepcopy(pointInCol)
                for point in pointInCol:
                    for route in routeList:
                        if len(actualPinC)>0 and pointInCol.index(point)<len(pointInCol) and  [point,curY] in route:
                            neighbor = pointInCol[pointInCol.index(point)-1]
                            if [neighbor,curY] in route and (abs(route.index([point,curY])-route.index([neighbor,curY]))==1):
                                del actualPinC[actualPinC.index(point)]
                if(len(actualPinC)<len(pointInCol)):
                    a = 1
                    
                for idx in range(len(actualPinC)):
                    point = actualPinC[idx]
                    for route in routeList:
                        if [point,curY] in route:
                            formerPoint = route[route.index([point,curY])-1]
                            latterPoint = route[(route.index([point,curY])+1)%len(route)]
                            diff_f = 0
                            diff_l = 0
                            if formerPoint[0]!=point and formerPoint[1]!=curY:
                                diff_f = (formerPoint[0]-point)/(formerPoint[1]-curY)
                            if latterPoint[0]!=point and latterPoint[1]!=curY:
                                diff_l = (point-latterPoint[0])/(curY-latterPoint[1])
                            if (diff_f>0 and diff_l<0)or(diff_f<0 and diff_l>0):
                                actualPinC.insert(idx,point)
                                idx = idx+1
                                #print("Cord ",point," Added into index No.",idx," in Y=",curY)
                
                if len(actualPinC)%2==0 and len(actualPinC)>0:
                    i = 0 
                    linePaintedFlag[curY] = 1
                    for curX in range(0,np.shape(img_ori)[1]):
                        if len(actualPinC)-1>i and actualPinC[i]<curX and actualPinC[i+1]>curX:
                            i = i+1
                        elif actualPinC[-1]<curX:
                            i = len(actualPinC)
                        if i%2==1:
                            mouseMaskd[curY,curX] = 255
        #Median filter
        kern = 3
        for curY in range(kern,np.shape(img_ori)[0]-kern): 
            if linePaintedFlag[curY]==0 and np.sum(linePaintedFlag[curY-kern:curY+kern+1])>=kern:
                if(linePaintedFlag[curY-1]==1):
                    mouseMaskd[curY,:] = mouseMaskd[curY-1,:] 
                else:
                    mouseMaskd[curY,:] = mouseMaskd[curY+1,:] 
              
        cv2.imwrite(imgName+"_Msk.png",mouseMaskd)
        if showImage == True:  
            cv2.imshow('Mask',cv2.resize(mouseMaskd,(0,0),fx = 0.5,fy = 0.5))
            cv2.waitKey(0)

    print("Filtering Circles out of the border...")
    filtCircList = copy.deepcopy(circListR)
    j = 0
    while j <len(filtCircList):
        c = filtCircList[j]
        if(c[1]>=np.shape(img_ori)[0]-c[2] or c[0]>=np.shape(img_ori)[1]-c[2] or c[1]<=c[2] or c[0]<=c[2]):
            filtCircList = np.delete(filtCircList,j,axis = 0)            
        elif(mouseMaskd[c[1],c[0]]==0 or img_divdil[c[1],c[0]]==0):
        
            #print("circle at ",c,"deleted.")
            filtCircList = np.delete(filtCircList,j,axis = 0)
        else:
            j = j+1
    #print(filtCircList)

    print("Filling the Hollow Regions(This May Take Long Time)...")    
    
    coverMask = np.zeros(np.shape(img_ori), np.uint8)
    areaMask = np.zeros(np.shape(img_ori), np.uint8)
    filtCircListR = []
    covRate = 0
    expandRdius = 2
    centre = calAvgRad(filtCircList)
    
    for circ in filtCircList:
        filtCircListR.append((circ[0],circ[1],circ[2],dist_Euc([circ[0],circ[1]],centre)))
        cv2.circle(coverMask,(circ[0],circ[1]),circ[2]+expandRdius,255,-1)
        cv2.circle(areaMask,(circ[0],circ[1]),circ[2],255,-1)
        covRate += 3.141592653*(circ[2]**2)
    covRate = 1-(np.sum(areaMask)/255)/covRate
    # print("All circles actual coverage: ",(np.sum(areaMask)/255))
    
    print("Center at:",[np.round(centre[1]),np.round(centre[0])])
    if showImage == True:  
        temImg = cv2.cvtColor(coverMask, cv2.COLOR_GRAY2BGR)
        cv2.circle(temImg,(filtCircList[2][0],filtCircList[2][1]),filtCircList[0][2],(0,255,0),-1)
        cv2.circle(temImg,(int(np.round(centre[0])),int(np.round(centre[1]))),7,(255,0,255),-1)
        cv2.imshow("old center",cv2.resize(temImg,(0,0),fx = 0.5,fy = 0.5))
        cv2.waitKey(0)
        
    if coverMask[int(centre[1]),int(centre[0])]==0:
        print("The Geometrical Center is not inside the particle, rearranging...")
        curIdx = 0
        minIdx = 0
        minDist = img_ori.shape[0]
        for circ in filtCircList:
            if dist_Euc([circ[0],circ[1]],centre)<minDist:
                minIdx = curIdx
                minDist = dist_Euc([circ[0],circ[1]],centre)               
            curIdx = curIdx+1
        minC =  filtCircList[minIdx]
        centre_old = copy.deepcopy(centre)
        centre = [minC[0]-(minC[0]-centre[0])*minC[2]/minDist,minC[1]-(minC[1]-centre[1])*minC[2]/minDist]
        print("New Centre Generated at:",[np.round(centre[1]),np.round(centre[0])])
        
        if showImage == True:  
            temImg = cv2.cvtColor(coverMask, cv2.COLOR_GRAY2BGR)
            cv2.circle(temImg,(filtCircList[1][0],filtCircList[1][1]),filtCircList[1][2],(0,255,255),-1)
            cv2.circle(temImg,(int(np.round(centre[0])),int(np.round(centre[1]))),7,(255,0,0),-1)
            cv2.circle(temImg,(int(np.round(centre_old[0])),int(np.round(centre_old[1]))),7,(0,255,0),-1)
            cv2.imshow("new center",cv2.resize(temImg,(0,0),fx = 0.5,fy = 0.5))
            cv2.waitKey(0)
    
    if showImage == True:  
        cv2.imshow('coverMask',cv2.resize(coverMask,(0,0),fx = 0.5,fy = 0.5))
        cv2.waitKey(0)
    filtCircListR = np.array(filtCircListR,dtype = dtype)
    filtCircListR=np.sort(filtCircListR, order='ri')
    
    
    for curX in range(0,np.shape(img_ori)[1]):
        for curY in range(0,np.shape(img_ori)[0]):
            if(mouseMaskd[curY,curX]==255 and img_ori[curY,curX]<=80):

                if coverMask[curY,curX]==0:
                    radSum = 0
                    radCnt = 0
                    for c in filtCircListR:
                        if dist_Euc([curX, curY], [c[0], c[1]]) < (256.0 - int(img_ori[curY, curX])):
                            radSum +=c[2]
                            radCnt +=1
                    if radCnt>0:
                        c_new = np.array((curX,curY,(int)(radSum/radCnt),0.0),dtype = dtype)
                        filtCircListR = np.r_[filtCircListR,c_new]
                        #print("circle ",c_new," added")
                        cv2.circle(coverMask,(curX,curY),(int)(radSum/radCnt)+expandRdius,255,-1)

    filtCircList = copy.deepcopy(filtCircListR)
    filtCircListR = []

    if showImage == True:  
        img_cent = cv2.cvtColor(img_ori,cv2.COLOR_GRAY2BGR)
        cv2.circle(img_cent,(int(centre[0]),int(centre[1])),5,(0,255,0),-1)
        cv2.imshow("cent",cv2.resize(img_cent,(0,0),fx = 0.7,fy = 0.7))
        cv2.waitKey(0)
        
    img_out = cv2.cvtColor(img_ori,cv2.COLOR_GRAY2BGR)
    for circ in filtCircList:
        filtCircListR.append((circ[0],circ[1],circ[2],dist_Euc([circ[0],circ[1]],centre)))
        cv2.circle(img_out,(circ[0],circ[1]),circ[2],(0,255,0),1)
        cv2.circle(img_out,(circ[0],circ[1]),1,(0,0,255),1) 
    cv2.imwrite(imgPath[0:-4]+"-tagged.png",img_out)
    filtCircListR = np.array(filtCircListR,dtype = dtype)
    filtCircListR=np.sort(filtCircListR, order='ri')
    print("Filling Finish! Start Calculating Parameters...")
    
    
    
    contours, hierarchy = cv2.findContours(coverMask//255,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    hull = cv2.convexHull(cnt)
    
    feretDia = 0
    feretPt0 = hull[0]
    feretPt1 = hull[1]
    for pt0 in hull:
        for pt1 in hull:
            if dist_Euc(pt1[0],pt0[0])>feretDia:
                feretPt0 = pt0[0]
                feretPt1 = pt1[0]
                feretDia = dist_Euc(pt1[0],pt0[0])
    posMaxDist = 0
    negMaxDist = 0
    for pt in hull:
        dist = dist_2Line(pt[0],feretPt0,feretPt1)
        if dist>0 and dist>posMaxDist:
            posMaxDist = dist
        elif dist<0 and dist<negMaxDist:
            negMaxDist = dist
    feretRate = feretDia/(posMaxDist-negMaxDist)
    roundness = 4*np.sum(coverMask)/(255*np.pi*feretDia**2)
    
    cvxArea = np.zeros(np.shape(img_ori), np.uint8)
    cv2.fillConvexPoly(cvxArea,hull, 255)
    print("SOOTAREA:",np.sum(coverMask/255),"\nCONVEXAREA:",np.sum(cvxArea/255))
    convexity = np.sum(coverMask)/np.sum(cvxArea)    
    
    print("Calculation Results:\n\tFeret Diameter: ",feretDia*pix2nm,"\n\tAspect Rate: ",feretRate,"\n\tRoundness: ",roundness,"\n\tConvexity: ",convexity)
    
    if showImage == True: 
        ie = cv2.cvtColor(areaMask, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(ie,contours,-1,(0,0,255),3)
        cv2.polylines(ie, [hull], True, (0, 255, 0), 2) 
        cv2.circle(ie,(feretPt0[0],feretPt0[1]),3,(255,0,0),-1)
        cv2.circle(ie,(feretPt1[0],feretPt1[1]),3,(255,0,0),-1)
        cv2.imshow("Convex Hull",ie)
        cv2.waitKey(0)

    
    
    sumR3r2 = 0
    sumR3 = 0
    sumR0 = 0
    RgList = []
    R0List = []
    avgR = 0
    for circ in filtCircListR:
        avgR += circ[2]
    avgR = avgR/len(filtCircListR)
    for i in range(len(filtCircListR)):
        circ = filtCircListR[i]
        sumR3r2+=(circ[2]/avgR)**3*circ[3]**2
        sumR3+=(circ[2]/avgR)**3
        sumR0 += circ[2]
        RgList.append(np.sqrt(sumR3r2/sumR3))
        R0List.append(sumR0/(i+1))
    xList = []
    yList = []
    for i in range(len(R0List)):
        xList.append(np.log10(RgList[i]/R0List[i]))
        yList.append(np.log10(i+1))
    [Df,A]= np.polyfit(xList[int(len(R0List)*9/len(R0List)):],yList[int(len(R0List)*9/len(R0List)):],1)
    linX = [xList[int(len(R0List)*9/len(R0List))],xList[-1]]
    linY = [A+linX[0]*Df,linX[1]*Df+A]
    print("\tDf: ",Df)
    plt.plot(xList,yList,"r,",linX,linY,"b-")
    plt.savefig(imgPath[0:-3]+"png")
    plt.cla()

    print("Landing Data into Table File...")
    lgrList = copy.deepcopy(xList)
    lgnList = copy.deepcopy(yList)
    xList = []
    yList = []
    rList = []
    riList = []
    indexList = []

    for c in filtCircListR:
        indexList.append(len(indexList)+1)
        xList.append(c[0])
        yList.append(c[1])
        rList.append(c[2])
        riList.append(c[3])

    df2 = pd.DataFrame({
        'Index': pd.Series(indexList),
        'x': pd.Series(xList),
        'y': pd.Series(yList),
        'radius(nm)': pd.Series([r*pix2nm for r in rList]),
        'ri(nm)': pd.Series([r*pix2nm for r in riList]),
        'Rg(nm)': pd.Series([r*pix2nm for r in RgList]),
        'R0(nm)': pd.Series([r*pix2nm for r in R0List]),
        'Log(Rg/R0)': pd.Series(lgrList),
        'Log(N)': pd.Series(lgnList),
        'Df': pd.Series([Df]),
        'covRate': pd.Series([covRate]),
        'ECD(nm)': pd.Series([np.sqrt(np.sum(areaMask)/255/np.pi)*2*pix2nm]),
        'ESD(nm)': pd.Series([2*R0List[-1]*np.power(filtCircListR.shape[0],1/3)*pix2nm]),
        'AR': pd.Series([feretRate]),
        'RN': pd.Series([roundness]),
        'CV': pd.Series([convexity]),
        'Aa(nm^2)': pd.Series([np.sum(areaMask)/255*pix2nm*pix2nm]),
        'Lmax(nm)':pd.Series([feretDia*pix2nm])
    })

    df2.to_csv(csvPath,index = False,columns=['Index','x','y','radius(nm)','ri(nm)','Rg(nm)','R0(nm)','Log(Rg/R0)','Log(N)','ECD(nm)','ESD(nm)','Aa(nm^2)','Lmax(nm)','AR','RN','CV','Df','covRate'],encoding = "UTF-8")

    print("Process Finished! Please Check the File. Press Any Key to Continue.")
    #cv2.waitKey(0)
    plt.close()
    cv2.destroyAllWindows()
    
print("All Pics are Processed.")
