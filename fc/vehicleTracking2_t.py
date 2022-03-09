from re import purge
import numpy as np
import json
import argparse
import cv2
import sys
import datetime
from datetime import timedelta
#import imutils
import dlib
import math
import time

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo

from kutlemerkezitakip import KutleMerkeziTakipClass
from takipedilenarac import TakipEdilenAracClass

import requests
import io
import matplotlib.pyplot as plt


## example usage -- python3 vehicleTracking2.py --cfg /home/acun/Documents/detectron2/configs/COCO-Detection/retinanet_R_101_FPN_3x.yaml --wts ptzModelOutputs_HalfRes_HalfAnchor/model_0124395.pth --input test.avi --output qwe3.avi
def parseArgs():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='config',
        help='cfg model file (/path/to/model_config.yaml)',
        default=model_zoo.get_config_file("COCO-Detection/retinanet_R_101_FPN_3x.yaml"),
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default="ptzModelOutputs_HalfRes_HalfAnchor/model_gun9b.pth",
        type=str
    )
    parser.add_argument(
        '--input',
        dest='input',
        help='FUll Path of the video File',
        default=None,
        type=str
    )
    parser.add_argument(
        '--output',
        dest='output',
        help='Output name of the video File',
        default="cam1Out.avi",
        type=str
    )
    parser.add_argument(
        '--confidence',
        dest='confidence',
        help='minimum probability to filter weak detections',
        default=0.5,
        type=float
    )
    parser.add_argument(
        '--nmsth',
        dest='nmsth',
        help='threshold of the NMS',
        default=0.3,  ##0.4
        type=float
    )
    parser.add_argument(
        '--httppost',
        dest='httppost',
        help='Send Post address, T or F',
        default="True",  ##0.4
        type=str
    )
    parser.add_argument(
        '--date',
        dest='receivedEpoch',
        help='Date and Time',
        default="1643414897",  ##0.4
        type=str
    )
    parser.add_argument(
        '--json',
        dest='jsonPathROI',
        help='json path for ROIs',
        default=None,  ##0.4
        type=str
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


class PTZTracking:

    predictor = None
    vid = None
    vidOut = None
    vidHeat = None

    frame = None
    copyFrame = None
    copyFrameCount = None
    copyFrameHeatMap = None

    copyFrameCountCombined = None

    heatMap = None
    heatMapFrame = None
    
    ##
    dlibpast = 0

    kmt = KutleMerkeziTakipClass(maksAracinGorunmedigiFrameSayisi=15, maksUzaklik=250)
    takipEdilenAraclar = {}

    roiPts = []
    countRioPts = []

    colors = [(0,0,0),(0,0,0), (255,0,0), (0,255,0),(0,0,0), (0,0,255),(0,0,0), (0,255,255)]
    validClassIds = [2, 3, 5, 7]

    vidH = 1080     ##
    vidW = 1920

    roiXmid = 0
    roiYmid = 0
    roiXcount = 0
    roiYcount = 0
    httppost = False

    edgemaxarr = np.zeros((4,2))
    edgeminarr = np.zeros((4,2))

    baslangicTarihi = datetime.datetime.now()


    kavsakAnalizSayimlar = np.array([[00, 1, 2, 3, 4],
                                     [1, 0, 0, 0, 0],
                                     [2, 0, 0, 0, 0],
                                     [3, 0, 0, 0, 0],
                                     [4, 0, 0, 0, 0]])

    H, W = 0, 0

    coolingTime = 300 # cooling every 3 sec. 25 FPS
    heating = 100
    heatMapMax = 3000 # get max value in 1 min max vlue artirilacak ve scaling degistirilerek daha gec isinma yapilabilir. 
    scalingMax = heatMapMax/255.0
    startFrame = 0

## init Fonksiyonlari
    def __init__(self, config=None, weights=None, confidence=None, nmsth=None):
        # class cagirildiginda modeli olusturur ve load eder.
        if config and weights:
            self.defineNetwork(config, weights, confidence, nmsth)
        print('PTZTracking init !')

    def str2bool(self, strin):
        return strin.lower() in ("yes", "True" ,"true", "T", "t", "1")

    def sendPayload(self, type, data, postt):
        gtrh = datetime.datetime.fromtimestamp(int(args.receivedEpoch))
        
        gonderilecekTarih = gtrh + timedelta(seconds=PTZTracking.startFrame/30)
        gonderilecekTarihs = gonderilecekTarih.strftime("%Y-%m-%dT%H:%M:%S+03:00")

        headers = {
        'Content-Type': 'application/json'
        }

        payload_temp = json.loads(data)
        payload_temp["Tarih"] = gonderilecekTarihs
        data = json.dumps(payload_temp)


        if type == "Root":
            url = "http://hybs.premierturk.com/HYS.WebApi/api/InfluxAdd_Root"
        elif type == "Total":
            url = "http://hybs.premierturk.com/HYS.WebApi/api/InfluxAdd_Total"


        if self.httppost:
            print(data)           
            try:
                response = requests.request("POST", url, headers=headers, data=data)
                print(response.text)
            except requests.exceptions.RequestException as e:  # This is the correct syntax
                print(e)
                time.sleep(5)

                              

        
    def readRoiFromFile(self, fileName):
        # klasorde ayni konumda olan json dosyasini okur. ROI bolgesi sadece araclarin gectigi yol bolgesinin poligon noktalarini tutmaktadir.
        # Yapi {"ROI": [[2, 1103], [4, 1515], [101, 1516],...}
        # tespit yapilacak tum alan, secmekle ugrasmak istemiyorsan dort kose noktrl gir #TODO
        #2592, 1520
        with open(fileName) as jsonRoi:
            self.roiPts = json.load(jsonRoi)
            #self.roiPts = np.array([self.roiPts['ROI']])
            self.roiPts = np.array([[1,1],[self.vidW,1],[self.vidW,self.vidH],[1,self.vidH]])

    def readCountRoisFromFile(self, fileName):
        # Araclarin uzerinden gectigi zaman sayilacagi polgonlari tutar. 
        # Yapi [{"Id": 1, "Points": [[523, 546], [527, 594], [410, 600]....]},
        #       {"Id": 2, "Points": [[791, 586], [740, 651], [550, 598],...]}]
        filename = args.jsonPathROI
        if args.jsonPathROI != None:
            fileName = filename

        idCounter = 0

        with open(fileName) as jsonRoi:
            self.countRioPts = json.load(jsonRoi)
            #edgemaxarr = np.zeros((len(self.countRioPts),2))
            #edgeminarr = np.zeros((len(self.countRioPts),2))
            cnt = 0
            xminR = 0
            xmaxR = 0
            yminR = 0
            ymaxR = 0
            xminROIid = 0
            xmaxROIid = 0
            yminROIid = 0
            ymaxROIid = 0
            for pts in self.countRioPts:
                idCounter += 1
                pts['intId'] = idCounter
                pts['Count'] = idCounter #0  # gecici !!!
                pts['CountG'] = 0
                pts['CountC'] = 0
                
                if pts['X'] == 1:
                    xmin = np.amin(np.array(pts['Points'])[:,0])
                    xmax = np.amax(np.array(pts['Points'])[:,0])
                    ymin = np.amin(np.array(pts['Points'])[:,1])
                    ymax = np.amax(np.array(pts['Points'])[:,1])
                    roiXmid =(xmin+xmax)/2 #= int(cv2.mean(np.array(pts['Points'])[:,0])[0])    ## veya =(xmin+xmax)/2
                    roiYmid =(ymin+ymax)/2 #= int(cv2.mean(np.array(pts['Points'])[:,1])[0])    ## veya =(ymin+ymax)/2
                    if xminR == 0 or xminR > roiXmid:     ## en alttaki ROI'yi bul
                        xminR = roiXmid
                        xminROIid = idCounter
                    if xmaxR == 0 or xmaxR < roiXmid:   ## en ustteki ROI'yi bul
                        xmaxR = roiXmid
                        xmaxROIid = idCounter

                    if yminR == 0 or yminR > roiYmid:     ## en sagdaki ROI'yi bul
                        yminR = roiYmid
                        yminROIid = idCounter
                    if ymaxR == 0 or ymaxR < roiYmid:   ## en soldaki ROI'yi bul
                        ymaxR = roiYmid
                        ymaxROIid = idCounter

                    xtemp = int(xmin + xmax / 2)
                    self.edgemaxarr[cnt] = np.array(pts['Points'])[np.argmax(np.array(pts['Points'])[:,0])]
                    self.edgeminarr[cnt] = np.array(pts['Points'])[np.argmin(np.array(pts['Points'])[:,0])]
                    self.roiXmid += int(cv2.mean(np.array(pts['Points'])[:,0])[0]) ## koselerin orta noktasi, 4ten fazla kose varsa sacmalayabiliyor
                    self.roiXmid += xtemp ## en uc iki noktanin ortasi
                    self.roiXcount += 1
                if pts['Y'] == 1:
                    xmin = np.amin(np.array(pts['Points'])[:,0])
                    ymin = np.amin(np.array(pts['Points'])[:,1])
                    xmax = np.amax(np.array(pts['Points'])[:,0])
                    ymax = np.amax(np.array(pts['Points'])[:,1])
                    roiXmid =(xmin+xmax)/2 #= int(cv2.mean(np.array(pts['Points'])[:,0])[0])    ## veya =(xmin+xmax)/2
                    roiYmid =(ymin+ymax)/2 #= int(cv2.mean(np.array(pts['Points'])[:,1])[0])    ## veya =(ymin+ymax)/2
                    if xminR == 0 or xminR > roiXmid:     ## en alttaki ROI'yi bul
                        xminR = roiXmid
                        xminROIid = idCounter
                    if xmaxR == 0 or xmaxR < roiXmid:   ## en ustteki ROI'yi bul
                        xmaxR = roiXmid
                        xmaxROIid = idCounter

                    if yminR == 0 or yminR > roiYmid:     ## en sagdaki ROI'yi bul
                        yminR = roiYmid
                        yminROIid = idCounter
                    if ymaxR == 0 or ymaxR < roiYmid:   ## en soldaki ROI'yi bul
                        ymaxR = roiYmid
                        ymaxROIid = idCounter
                        

                    ytemp = int(ymin + ymax / 2)
                    self.edgemaxarr[cnt] = np.array(pts['Points'])[np.argmax(np.array(pts['Points'])[:,1])]
                    self.edgeminarr[cnt] = np.array(pts['Points'])[np.argmin(np.array(pts['Points'])[:,1])]
                    self.roiYmid += int(cv2.mean(np.array(pts['Points'])[:,1])[0])  ##  koselerin orta noktasi, 4ten fazla kose varsa sacmalayabiliyor
                    self.roiYmid += ytemp   ## en uc iki noktanin ortasi
                    self.roiYcount += 1
                cnt += 1

        pass


        if self.roiXcount != 0:
            self.roiXmid = int (self.roiXmid / self.roiXcount)
        if self.roiYcount != 0:
            self.roiYmid = int (self.roiYmid / self.roiYcount)

        self.roiXmid = int ((np.amax(self.edgemaxarr[:,0]) + np.amin(self.edgeminarr[:,0]))/2) 
        self.roiYmid = int ((np.amax(self.edgemaxarr[:,1]) + np.amin(self.edgeminarr[:,1]))/2) 
        self.roiXmid = 900
        self.roiYmid = 410
        

    def defineNetwork(self, configPath, weights, confidence, nms):
        # detectron2 ile egitilmis modelin test asamasnda gerekli parametreleri ayarlaniyor.
        cfg = get_cfg()
        cfg.merge_from_file(configPath)

        cfg.OUTPUT_DIR = ''
        cfg.MODEL.RETINANET.NUM_CLASSES = 80
        cfg.MODEL.RETINANET.NMS_THRESH_TEST = nms
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence
        #cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8, 16, 32, 64, 256]]
        cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16, 32, 64, 128, 256]]

        cfg.INPUT.MIN_SIZE_TEST = 760
        cfg.INPUT.MAX_SIZE_TEST = 1296
        
        cfg.MODEL.WEIGHTS = weights
        self.predictor = DefaultPredictor(cfg)
        print('detector init !')

    def multiClassNMS(self, logname, boxes, confidences, classIDs, vis=False):
        isFound = np.zeros(len(boxes), int)
        trueDets = []
        trueBoxes = []
        trueConf = []
        trueClassIds = []
        minDist = 150 #8
        # alpha = .8


        if len(logname) < 3:
            write_log = False
        else:
            write_log = True

        
        for bb in range(0, len(boxes)):
            if isFound[bb] == 1: continue
        
            midPointDet = [int(boxes[bb][0]) + int(boxes[bb][2] / 2), int(boxes[bb][1]) + int(boxes[bb][3] / 2)] #x1 + x2/2 , y1 + y2/2
            closeBoxes = []
            

            distlogname = logname[:len(logname)-4] + "_dists.txt"

            if write_log:
                tempText = "Frame = :" + str(PTZTracking.startFrame) + "\n"
                with open(distlogname, "a") as logtxt:
                    logtxt.write(tempText)



            for cc in range(0, len(boxes)):
                if isFound[cc] == 1: continue
            
                midPointDet2 = [int(boxes[cc][0]) + int(boxes[cc][2] / 2), int(boxes[cc][1]) + int(boxes[cc][3] / 2)]

                dist = math.sqrt(((midPointDet[0] - midPointDet2[0]) ** 2) + ((midPointDet[1] - midPointDet2[1]) ** 2))

                if write_log:
                    tempText = "\n 1. box: " + str(boxes[bb]) + ", midpoint: " + str(midPointDet) + ", 2. box: " + str(boxes[cc]) + ", midpoint: " + str(midPointDet2)
                    tempText+= ", diff dist: " + str(dist) + ", minCheckDist: " + str(minDist)
                    with open(distlogname, "a") as logtxt:
                        logtxt.write(tempText)

                if dist < minDist:
                    closeBoxes.append(cc)
                    isFound[cc] = 1

            
        
            if closeBoxes:

                scores = [confidences[sub] for sub in closeBoxes]
                # bbs = [boxes[sub] for sub in closeBoxes]
                # area = [bbs[i][2] * bbs[i][3] for i in range(0, len(bbs))]
                # idx = np.argmax(np.array(area) + (np.array(scores) * alpha))
                idx = np.argmax(np.array(scores))

                trueBoxes.append(boxes[closeBoxes[idx]])
                trueConf.append(confidences[closeBoxes[idx]])
                trueClassIds.append(classIDs[closeBoxes[idx]])

        if True:    ## bboxlar kutu cizdirme kapatildi.
            for bb in range(len(trueBoxes)):

                text = '{}: {:.4f}'.format(trueClassIds[bb], trueConf[bb])
                x, y = trueBoxes[bb][0], trueBoxes[bb][1]
                w, h = trueBoxes[bb][0] + trueBoxes[bb][2], trueBoxes[bb][1] + trueBoxes[bb][3] 
                cv2.rectangle(self.copyFrame,(x, y),(w, h), self.colors[trueClassIds[bb]], 2)       ##bboxlar cizdirme
                cv2.putText(self.copyFrame, text, (x, int(y-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors[trueClassIds[bb]], 2)
                

        
        return trueBoxes, trueConf, trueClassIds

## Video Fonksiyonlari
    def vidCapture(self, input):
        # Girilen input videosunu okur ve gerekli bilgileri elde
        self.vid = cv2.VideoCapture(input)
        try:
            prop = cv2.CAP_PROP_FRAME_COUNT
            total = int(self.vid.get(prop))
            print("[INFO] {} total frames in video".format(total))
            self.W  = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.H = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
        except:
            print("[INFO] could not determine # of frames in video")
            print("[INFO] no approx. completion time can be provided")
            total = -1

    def createVideoWriter(self, fileName, fps=25.0, dims=(2592,1520)):
        # Tespit sonuclarini keyetmek icin yeni bir output videosu olusturmaktadr.
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        vid = cv2.VideoWriter(fileName, fourcc, fps, dims)
        return vid 

    def writeVideo(self,vid, frame):
        # Elde edilen sonuc framelerini olusturulan cikis videosuna yazar
        vid.write(frame)

    def releaseVideo(self):
        # Program sonlandiginda olusturulan videoyu kapatir.
        self.vidHeat.release()


## Hareketli araclarin heatmaplarinin olusturulmasi
    def plotHeatmapLine(self, pts):
        pts = np.array(pts)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(self.heatMapFrame, [pts], False, self.heating, 30)  #25 idi
                
    def coolHeatmap(self, fr):
        if fr % self.coolingTime == 0: 
            print('Current Frame=', fr)
            self.heatMap = np.multiply(self.heatMap, np.where(self.heatMap>1,0.95,1))
            self.heatMap = self.heatMap.astype(np.uint16)            
        
    def plotHeatmap(self):
        
        self.heatMap = cv2.add(self.heatMap, self.heatMapFrame)
        
        self.heatMap[self.heatMap > self.heatMapMax] = self.heatMapMax
        self.heatMap = cv2.GaussianBlur(self.heatMap,(15,15),sigmaX=1.5,sigmaY=1.5)    
        heatMapUint8= cv2.convertScaleAbs(self.heatMap/self.scalingMax)
        heatMatColor = cv2.applyColorMap(heatMapUint8, cv2.COLORMAP_HOT) # COLORMAP_HOT
        self.copyFrameHeatMap = cv2.addWeighted(self.copyFrame, 0.7, heatMatColor, 0.3, 0)

    def plotHeatmapandCounts(self):
        
        self.heatMap = cv2.add(self.heatMap, self.heatMapFrame)
        
        self.heatMap[self.heatMap > self.heatMapMax] = self.heatMapMax
        self.heatMap = cv2.GaussianBlur(self.heatMap,(1,1),sigmaX=0.55,sigmaY=0.55)    
        heatMapUint8= cv2.convertScaleAbs(self.heatMap/self.scalingMax)
        heatMatColor = cv2.applyColorMap(heatMapUint8, cv2.COLORMAP_JET) # COLORMAP_HOT
        self.copyFrameHeatMap = cv2.addWeighted(self.copyFrame, 0.75, heatMatColor, 0.25, 0)
        
        im = self.copyFrameHeatMap # self.copyFrame
        countPts = self.countRioPts 

        copyIm = im.copy()
        for pts in countPts:
            #text = '{}'.format(pts['Count'])
            #ptsx, ptsy = pts['Points'][0]
            #cv2.putText(copyIm, text, (ptsx, ptsy-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0),3)
            cv2.fillPoly(copyIm, np.int32([pts['Points']]), (0,255,0))
        self.copyFrameCountCombined = cv2.addWeighted(copyIm, 0.4, im, 0.6, 0)
        '''
        cv2.circle(self.copyFrameCountCombined,(self.roiXmid,self.roiYmid),4,(255,255,255),4)
        cv2.circle(self.copyFrameCountCombined,(int(self.edgemaxarr[0][0]),int(self.edgemaxarr[0][1])),2,(0,0,255),2)
        cv2.circle(self.copyFrameCountCombined,(int(self.edgemaxarr[1][0]),int(self.edgemaxarr[1][1])),3,(0,0,255),3)
        cv2.circle(self.copyFrameCountCombined,(int(self.edgemaxarr[2][0]),int(self.edgemaxarr[2][1])),4,(0,0,255),4)
        cv2.circle(self.copyFrameCountCombined,(int(self.edgemaxarr[3][0]),int(self.edgemaxarr[3][1])),5,(0,0,255),5)
        cv2.circle(self.copyFrameCountCombined,(int(self.edgeminarr[0][0]),int(self.edgeminarr[0][1])),2,(0,255,255),2)
        cv2.circle(self.copyFrameCountCombined,(int(self.edgeminarr[1][0]),int(self.edgeminarr[1][1])),3,(0,250,255),3)
        cv2.circle(self.copyFrameCountCombined,(int(self.edgeminarr[2][0]),int(self.edgeminarr[2][1])),4,(0,250,255),4)
        cv2.circle(self.copyFrameCountCombined,(int(self.edgeminarr[3][0]),int(self.edgeminarr[3][1])),5,(0,250,255),5)
        '''
        for pts in countPts:
            text = '{}'.format(pts['Count'])
            ptsx, ptsy = pts['Points'][0]
            cv2.putText(self.copyFrameCountCombined, text, (ptsx, ptsy-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255),3)
            #cv2.putText(self.copyFrameCountCombined, str(pts['Id']), (ptsx-30, ptsy+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255),3)


## Araclarin belirli yerlerden gecis sayimi
    def plotRoisOnImage(self,im, countPts):
        # araclarin uzerinden gectigi zaman sayilacagi polygon alanlari imge uzerinde cizdirir.
        copyIm = im.copy()
        for pts in countPts:
            text = '{}'.format(pts['Count'])
            ptsx, ptsy = pts['Points'][0]
            cv2.putText(copyIm, text, (ptsx, ptsy-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0),3)
            cv2.fillPoly(copyIm, np.int32([pts['Points']]), (0,255,0))
        self.copyFrameCount = cv2.addWeighted(copyIm, 0.45, im, 0.55, 0)
        
    def counter(self, logname, takip_edilen_arac):
        # Takip edilen noktalar herhangi bir sayim poligonuna dustugu 
        # zaman ilgili poligonun sayim degerini artiran fonksiyon

        # inRois = False
        incounter = 0
        if len(logname) < 3:
            write_log = False
        else:
            write_log = True

        point = takip_edilen_arac.kutle_merkezleri[-1]
        idText = str(takip_edilen_arac.sinif) + str(takip_edilen_arac.aracID) 
        cv2.putText(self.copyFrame, idText, (point[0], point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,150,255), 2) # aracID arac ustune yaz
        if write_log:
            tempText = str(incounter) + ". aracID:" + idText + ", "
            incounter += 1
            with open(logname, "a") as logtxt:
                logtxt.write(tempText)


        for ptz in self.countRioPts:

            km = takip_edilen_arac.kutle_merkezleri
            kmCount = len(km)
            if  kmCount < 2: 
                continue

            ## TODO: Frame merkez konumunu magic number girme!
            CenterX = 1920/2
            CenterY = 1080/2

            if kmCount > 5:
                kmLim = 4
            else:
                kmLim = kmCount ## 3 veya 2 olabilir

            roi = np.array(ptz['Points'])
            isInPolygon = cv2.pointPolygonTest(roi, point , False)
            if isInPolygon > 0:
                
                roiMidX = int(cv2.mean(roi[:,0])[0])
                roiMidY = int(cv2.mean(roi[:,1])[0])

                CenterX = self.roiXmid
                CenterY = self.roiYmid

                deltaX = km[kmCount-1][0] - km[kmCount-kmLim][0] ## pozitif sonuc = gecis yonu sag
                deltaY = km[kmCount-1][1] - km[kmCount-kmLim][1] ## pozitif sonuc = gecis yonu asagi

                girisX = False
                if (deltaX > -3 and CenterX > km[kmCount-1][0]) or (deltaX < 3 and CenterX < km[kmCount-1][0]):
                    ## saga gidiyor ve kadrajda solda       veya       sola gidiyor ve kadrajda sagda
                    girisX = True              ## yatay eksene gore giris yapmis
                
                girisY = False
                if (deltaY > -3 and CenterY > km[kmCount-1][1]) or (deltaY < 3 and CenterY < km[kmCount-1][1]):
                    ## asagi gidiyor ve kadrajda yukarida       veya       yukari gidiyor ve kadrajda asagida
                    girisY = True              ## dusey eksene gore giris yapmis
                

                directionX = False
                if abs(deltaX) > abs(deltaY):   ## Yatay eksendeki fark dusey eksendeki farktan daha buyukse
                    directionX = True           ## arac asil yatay eksende hareket ediyor demektir

                #print("ROI ID=" + str(ptz['Id']) + ", x=" + str(ptz['X']) + ", y=" + str(ptz['Y']))
                #print("xdif = " + str(deltaX) + ",ydif = " + str(deltaY) + ",yatay = " + str(directionX) + ",Xgiris = " + str(girisX) + ",Ygiris = " + str(girisY))
                
                postt = False
                girisTuru = "Cikis"
                toplamRoi = 0

                if takip_edilen_arac.aracID == 16:
                    sfasf = 3

                if ptz['X'] == 1:                               ## ROI yatay eksen ile ilgili sayim yapiyor
                    if girisX:                                  ## aracin hareket yonu giris mi
                        if takip_edilen_arac.ilkGecis == -1:    ## ilk giris
                            takip_edilen_arac.ilkGecis = ptz['Id']
                            ptz['Count'] += 1 
                            ptz['CountG'] += 1 
                            postt = True
                            girisTuru = "Giris"
                            toplamRoi = ptz['CountG']
                    else:
                        if takip_edilen_arac.ikinciGecis == -1 and takip_edilen_arac.ilkGecis != ptz['Id']:     ## cikis yapmis
                            takip_edilen_arac.ikinciGecis = ptz['Id']
                            ptz['Count'] += 1 
                            ptz['CountC'] += 1 
                            postt = True
                            toplamRoi = ptz['CountC']
                            if takip_edilen_arac.ilkGecis == -1:    ## cikis yapan aracin girisi o aracla iliskilendirilememis
                                takip_edilen_arac.ilkGecis = 0

                elif ptz['Y'] == 1:                               ## ROI dusey eksen ile ilgili sayim yapiyor
                    if girisY:                                  ## aracin hareket yonu giris mi
                        if takip_edilen_arac.ilkGecis == -1:    ## ilk giris
                            takip_edilen_arac.ilkGecis = ptz['Id']
                            ptz['Count'] += 1
                            ptz['CountG'] += 1  
                            girisTuru = "Giris"
                            toplamRoi = ptz['CountG']
                            postt = True
                    else:
                        if takip_edilen_arac.ikinciGecis == -1 and takip_edilen_arac.ilkGecis != ptz['Id']:     ## cikis yapmis
                            takip_edilen_arac.ikinciGecis = ptz['Id']
                            ptz['Count'] += 1 
                            ptz['CountC'] += 1 
                            toplamRoi = ptz['CountC']
                            postt = True
                            if takip_edilen_arac.ilkGecis == -1:    ## cikis yapan aracin girisi o aracla iliskilendirilememis
                                takip_edilen_arac.ilkGecis = 0


                '''
                elif takip_edilen_arac.ilkGecis == -1:
                    takip_edilen_arac.ilkGecis = ptz['Id']
                    ptz['Count'] += 1   
                elif takip_edilen_arac.ikinciGecis == -1 and takip_edilen_arac.ilkGecis != ptz['Id']:
                    takip_edilen_arac.aracSayildiRoi = ptz['Id']
                    takip_edilen_arac.ikinciGecis = ptz['Id']
                    ptz['Count'] += 1 
                '''

                gonderilecekTarih = self.baslangicTarihi + timedelta(seconds=PTZTracking.startFrame/30)
                gonderilecekTarihs = gonderilecekTarih.strftime("%Y-%m-%dT%H:%M:%S+03:00")

                payload = json.dumps({
                "id": 11,
                "Tur": girisTuru,
                "Loop": str(ptz['Id']),
                "Adet": toplamRoi
                })

                
                if postt and self.httppost:
                    #PTZTracking.sendPayload(self, "Total", payload, postt)
                    postt = False
                                
                if len(logname) < 3:
                    write_log = False
                else:
                    write_log = True
                    postlogname = logname[:len(logname)-4] + "postTotal.txt"

                if write_log and postt:
                    with open(postlogname, "a") as logtxt:
                        logtxt.write(payload)
                        logtxt.write("\n")


## Process Fonksiyonlari
    def runDetection(self, frame, logname, fr,  pred):
        # Yuklenen model ile gelen frame uzerinde tespit yapan fonksiyon.

        boxes = []
        confidences = []
        classIDs = []
        
        incounter = 0
        if len(logname) < 3:
            write_log = False
        else:
            write_log = True
            with open(logname, "a+") as logtxt:
                logtxt.write("\n\n New Frame started \n")

                
        outputs = pred(fr)
        for bb in range(len(outputs["instances"].scores)):

            pred = outputs["instances"].to("cpu")  ##cuda???
            clsId = pred.pred_classes[bb].item()

            if write_log:
                tempText = str(incounter) + ". *_frame: " + str(PTZTracking.startFrame) + "_*, class = " + str(clsId) + ", "
                with open(logname, "a") as logtxt:
                    logtxt.write(tempText)
                    incounter += 1

            if clsId in  self.validClassIds:

                bbox = pred.pred_boxes[bb].tensor.numpy()[0] ## bbox xmin,ymin,xmax,ymax
                score =  pred.scores[bb].item()

                boxes.append([int(bbox[0]), int(bbox[1]), int(bbox[2]-bbox[0]), int(bbox[3]-bbox[1])])
                confidences.append(float(score))
                classIDs.append(clsId)
                if write_log:
                    tempText = "??, class = " + str(clsId) + ", score = " + str(score) + ", bbox = " + str(bbox) +"\n" 
                    with open(logname, "a") as logtxt:
                        logtxt.write(tempText)


        return boxes, confidences, classIDs
        
    def trackDetections(self, frame, boxes, classIDs, roiPts):
        dikdortgen = []
        takipEdiciler = []
        dlibpast = dlib.correlation_tracker()
        for bb in boxes:
            x, y = bb[0], bb[1]
            w, h = bb[2], bb[3]
            isInPolygon = cv2.pointPolygonTest(roiPts, (int(x+w/2),int(y+h/2)), False)
            if isInPolygon<1: continue
        
            bbox = [int(x), int(y), int(x+w), int(y+h)]
            dlibKorelasyonTakipEdici = dlib.correlation_tracker()
            classID = 1 # classIDs[i]
            dikdortgen = dlib.rectangle(bbox[0], bbox[1], bbox[2], bbox[3])
            dlibKorelasyonTakipEdici.start_track(frame, dikdortgen)
            takipEdiciler.append([dlibKorelasyonTakipEdici, classID])
            notNull = True
        if len(takipEdiciler) == 0:
            dlibKorelasyonTakipEdici = dlibpast
            notNull = False
        return takipEdiciler, dlibKorelasyonTakipEdici, notNull       
    
    def updateTrackPoss(self, frame, takipEdiciler, dlibKorelasyonTakipEdici):
        dikdortgenler = []
        for i in takipEdiciler:
            dlibKorelasyonTakipEdici = i[0]
            dlibKorelasyonTakipEdici.update(frame)
            konum = dlibKorelasyonTakipEdici.get_position()
            baslangicX = int(konum.left())
            baslangicY = int(konum.top())
            bitisX = int(konum.right())
            bitisY = int(konum.bottom())
            dikdortgenler.append([(baslangicX, baslangicY, bitisX, bitisY), i[1]])
        return dikdortgenler

    def processTracks(self, logname, araclar, takipEdilenAraclar, aracSiniflari):
        # Takip edilen noktalarin yeni gelen noktalarini ilgili nesneye ekleyerek sayim ve heatmap'leri cikartan fonksiyon.
        
        for (aracID, kutle_merkezi) in araclar.items():
            takip_edilen_arac = takipEdilenAraclar.get(aracID, None)
            arac_sinif = aracSiniflari[aracID]
            
            if takip_edilen_arac is None:
                takip_edilen_arac = TakipEdilenAracClass(aracID, kutle_merkezi, arac_sinif)
            else:   
                takip_edilen_arac.kutle_merkezleri.append(kutle_merkezi) 
            takipEdilenAraclar[aracID] = takip_edilen_arac
            
            self.counter(logname, takip_edilen_arac) # Arac Sayim Fonksiyonu   
        self.plotRoisOnImage(self.copyFrame, self.countRioPts)

        # takipEdilenAraclar listesinde olusturulmus ve goruntuden cikip artik takibi olmayan nesneleri temizleyen kisim.     
        keys2Remove = []
        for key in takipEdilenAraclar:
            if key in araclar.keys(): continue
            keys2Remove.append(key)
            self.plotHeatmapLine(takipEdilenAraclar[key].kutle_merkezleri)
        self.plotHeatmapandCounts() ## self.plotHeatmap()   ### kapatildi, eski hale getirmek icin acilabilir

        for key in keys2Remove:     ## silinen araclar burada siliniyor ANALIZ BURADA KAVSAK 2022

            if takipEdilenAraclar[key].ilkGecis != -1:
                distlogname = logname[:len(logname)-4] + "_gecisler.txt"
                tempText = "aracID = " + str(takipEdilenAraclar[key].sinif) + str(takipEdilenAraclar[key].aracID) + " ilk = " + str(takipEdilenAraclar[key].ilkGecis) + " ikinci = " + str(takipEdilenAraclar[key].ikinciGecis) + ", frame =  " + str(PTZTracking.startFrame) + "\n"
                #with open(distlogname, "a") as logtxt:
                #    logtxt.write(tempText)
            
            if takipEdilenAraclar[key].ikinciGecis != -1:
                distlogname = logname[:len(logname)-4] + "_gecisler_kusursuz.txt"
                tempText = "aracID = " + str(takipEdilenAraclar[key].sinif) + str(takipEdilenAraclar[key].aracID) + " ilk = " + str(takipEdilenAraclar[key].ilkGecis) + " ikinci = " + str(takipEdilenAraclar[key].ikinciGecis) +  ", frame =  " + str(PTZTracking.startFrame) + "\n"
                #with open(distlogname, "a") as logtxt:
                #   logtxt.write(tempText)
                #print("ilk = " + str(takipEdilenAraclar[key].ilkGecis))
                #print("ikinci = " + str(takipEdilenAraclar[key].ikinciGecis))
                #self.kavsakAnalizSayimlar[takipEdilenAraclar[key].ilkGecis][takipEdilenAraclar[key].ikinciGecis] += 1
                #print(self.kavsakAnalizSayimlar)

                distlogname = logname[:len(logname)-4] + "_gecisler_kusursuz_array.txt"
                #with open(distlogname, "w") as logtxt:
                #    logtxt.write("satirlar giris, sutunlar cikis" + "\n")
                #    logtxt.write("ilk satir ve sutunlar girisleri belirtiyorlar" + "\n" + "\n")
                #    for line in self.kavsakAnalizSayimlar:
                #        logtxt.write(" ".join(str(line)) + "\n")
                
                
                payload = json.dumps({
                "G_Loop": str(takipEdilenAraclar[key].ilkGecis),
                "C_Loop": str(takipEdilenAraclar[key].ikinciGecis)
                })

                if self.httppost and takipEdilenAraclar[key].ilkGecis != 0:
                    #PTZTracking.sendPayload(self, "Root", payload, True)
                    asfasf=3
                
                if len(logname) < 3:
                    write_log = False
                else:
                    write_log = True
                    postlogname = logname[:len(logname)-4] + "postRoot.txt"

                if write_log:
                    with open(postlogname, "a") as logtxt:
                        logtxt.write(payload)
                        logtxt.write("\n")


            del takipEdilenAraclar[key]
            
        return takipEdilenAraclar

    def runFrames(self, logname, httppostArg, vis):
        # Video uzerinde her bir frame'i modele sokup gerekli takip ve sayim fonksiyonlarini cagiran ana dongu.
        self.heatMap = np.zeros((self.H, self.W), np.uint16)
        fr = 0
        self.httppost = httppostArg
        onetime=False

        while True:

            grabbed, self.frame = self.vid.read()
            if not grabbed: break
            t1 = time.time()

            if onetime and self.httppost:
                #do http post
                

                gonderilecekTarih = self.baslangicTarihi + timedelta(seconds=PTZTracking.startFrame/30)
                if gonderilecekTarih.second == 0:
                    gonderilecekTarih + timedelta(11)
                gonderilecekTarihs = gonderilecekTarih.strftime("%Y-%m-%dT%H:%M:%S+03:00")

                url = "http://10.210.210.95:2101/WebApi/api/InfluxAdd_Total"
                payload = json.dumps({
                "id": 11,
                "Tur": "Giris",
                "Loop": "3",
                "Adet": 8,
                "Tarih": gonderilecekTarihs
                })
                headers = {
                'Content-Type': 'application/json'
                }

                #response = requests.request("POST", url, headers=headers, data=payload)

                #print(response.text)

                onetime = False


            fr+=1
            framTxt = str(fr) + ', model9b' #model sol ust yazi text
            cv2.putText(self.frame, framTxt, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,250,255), 2) # framecounter
            #if fr > 25*60*5:
            #if fr > 70000:
            #    self.releaseVideo()
            #    break
            #if fr < 132765:
            #    continue
            if fr > 430:
                control = 1
            #if fr % 3 == 0:
            #    continue

            PTZTracking.startFrame = fr
            #if fr%10 == 0:
                #print('Frame -- {}'.format(fr))
            self.copyFrame = self.frame.copy()
            self.heatMapFrame = np.zeros((self.H, self.W), np.uint16)
            
            boxes, confidences, classIDs = self.runDetection(fr, logname, self.frame, self.predictor)
            boxes, confidences, classIDs = self.multiClassNMS(logname, boxes, confidences, classIDs, vis)
            t2 = time.time()
            t3 = time.time()
            t4 = time.time()
            t5 = time.time()
            t6 = time.time()
            t7 = time.time()

            takipEdiciler, dlibKorelasyonTakipEdici, notNull = self.trackDetections(self.frame, boxes, classIDs, self.roiPts) 
            if notNull:
                t3 = time.time()
                dikdortgenler = self.updateTrackPoss(self.frame, takipEdiciler, dlibKorelasyonTakipEdici)
                t4 = time.time()                
                araclar, aracSiniflari = self.kmt.guncelle(dikdortgenler)
                self.coolHeatmap(fr)
                t5 = time.time()
                self.takipEdilenAraclar = self.processTracks(logname, araclar, self.takipEdilenAraclar, aracSiniflari)
                t6 = time.time()
            
            ## Sayim videosunu kaydeder.
            #self.writeVideo(self.vidOut, self.copyFrameCount)
            ## Heatmap videosunu kaydeder.
            #self.writeVideo(self.vidHeat, self.copyFrameHeatMap)
            
            self.writeVideo(self.vidHeat, self.copyFrameCountCombined)
            t7 = time.time()
            print(str(t2-t1) + ", " + str(t3-t2) + ", "  + str(t4-t3) + ", "  + str(t5-t4) + ", "  + str(t6-t5) + ", " + str(t7-t6))

        self.releaseVideo()
        
        return


def main(args):
    print("\n\n\n")
    print(args)
    print("\n\n\n")
    #with open("started11111111.txt", "a") as logtxt:
    #    logtxt.write(str(args))
    time.sleep(1)

    ptzTracker = PTZTracking(args.config, args.weights, args.confidence, args.nmsth)
    #ptzTracker.vidOut = ptzTracker.createVideoWriter(args.output, fps=30.0, dims=(1920,1080))
    ptzTracker.vidHeat = ptzTracker.createVideoWriter('heatMap.avi', fps=30.0, dims=(1920,1080))
    logname = "log_" + args.output + "_" + str(datetime.datetime.now().hour) + "." + str(datetime.datetime.now().minute) + ".txt"
    logname ="no"
    ptzTracker.readRoiFromFile('selectedRoi.json')
    ptzTracker.readCountRoisFromFile('countRois.json')
    ptzTracker.vidCapture(args.input)
    ptzTracker.runFrames(logname, ptzTracker.str2bool(args.httppost), vis=True)

    
if __name__ == '__main__':

    args = parseArgs()
    main(args)

