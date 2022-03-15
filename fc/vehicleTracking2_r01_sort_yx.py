from re import purge
import numpy as np
import json
import argparse
import cv2
import sys
import datetime
from datetime import timedelta
import dlib
import math
import time

import torch
from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis



from kutlemerkezitakip import KutleMerkeziTakipClass
from takipedilenarac import TakipEdilenAracClass

import requests
import io
import matplotlib.pyplot as plt

from sort_wc_nd import *

### YOLOX CLASS
class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=COCO_CLASSES,
        decoder=None,
        device="gpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)


    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            #print("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res

###



## example usage -- python3 vehicleTracking2.py --cfg /home/acun/Documents/detectron2/configs/COCO-Detection/retinanet_R_101_FPN_3x.yaml --wts ptzModelOutputs_HalfRes_HalfAnchor/model_0124395.pth --input test.avi --output qwe3.avi
def parseArgs():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='config',
        help='cfg model file (/path/to/model_config.yaml)',
        default="old",##model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"),
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default="/home/acun/Desktop/PTZCamera/vehicleTracking/ptzModelOutputs_HalfRes_HalfAnchor/model_gun9b.pth",
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
        default="camoutDefault.avi",
        type=str
    )
    parser.add_argument(
        '--confidence',
        dest='confidence',
        help='minimum probability to filter weak detections',
        default=0.25,
        type=float
    )
    parser.add_argument(
        '--nmsth',
        dest='nmsth',
        help='threshold of the NMS',
        default=0.45,  ##0.4
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
    parser.add_argument(
        '--xname',
        dest='xname',
        help='YOLOX name, eg yolox-x',
        default="yolox-x",
        type=str
    )
    parser.add_argument(
        '--tsize',
        dest='tsize',
        help='tsize for inference',
        default=640,
        type=int
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


class PTZTracking:

    mot_tracker = Sort()
    mot_memory = {}

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

    vidH = 1080     ## TODO videodan çek
    vidW = 1920

    roiXmid = 0
    roiYmid = 0
    roiXcount = 0
    roiYcount = 0
    httppost = False

    baslangicTarihi = datetime.datetime.now()
    vidid = int(baslangicTarihi.strftime("%y%m%d%H%M"))

    outputfolder = ""


    kavsakAnalizSayimlar = np.array([[00, 1, 2, 3, 4],
                                     [1, 0, 0, 0, 0],
                                     [2, 0, 0, 0, 0],
                                     [3, 0, 0, 0, 0],
                                     [4, 0, 0, 0, 0]])

    H, W = 0, 0

    coolingTime = 300 # cooling every 3 sec. 25 FPS
    heating = 100
    heatMapMax = 3000 # get max value in 1 min max vlue artırılarak ve scaling degistirilerek daha gec isinma yapilabilir.
    scalingMax = heatMapMax/255.0
    startFrame = 0

## init Fonksiyonlari
    def __init__(self, config=None, weights=None, confidence=None, nmsth=None):
        # class cagirildiginda modeli oluşturur ve load eder.
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
            url = "http://aus.kocaeli.bel.tr/WebApi/api/InfluxAdd_Root"
        elif type == "Total":
            url = "http://hybs.premierturk.com/HYS.WebApi/api/InfluxAdd_Total"
            url = "http://aus.kocaeli.bel.tr/WebApi/api/InfluxAdd_Total"

        self.httppost = False
        if self.httppost:
            print(data)
            response = requests.request("POST", url, headers=headers, data=data)
            print(response.text)
                              

        
    def readRoiFromFile(self, fileName):
        # klasörde aynı konumda olan json dosyasını okur. ROI bölgesi sadece araçların geçtiği yol bölgesinin poligon noktalarını tutmaktadır.
        # Yapı {"ROI": [[2, 1103], [4, 1515], [101, 1516],...}
        # tespit yapılacak tüm alan, seçmekle uğraşmak istemiyorsan dör köşe noktalarını gir #TODO
        #2592, 1520
        with open(fileName) as jsonRoi:
            self.roiPts = json.load(jsonRoi)
            #self.roiPts = np.array([self.roiPts['ROI']])
            self.roiPts = np.array([[1,1],[self.vidW,1],[self.vidW,self.vidH],[1,self.vidH]])

    def readCountRoisFromFile(self, fileName):
        # Araçların üzerinden geçtiği zaman sayılacağı polgonları tutar. 
        # Yapı [{"Id": 1, "Points": [[523, 546], [527, 594], [410, 600]....]},
        #       {"Id": 2, "Points": [[791, 586], [740, 651], [550, 598],...]}]
        filename = args.jsonPathROI
        if args.jsonPathROI != None:
            fileName = filename
        
        idCounter = 0

        with open(fileName) as jsonRoi:
            self.countRioPts = json.load(jsonRoi)
            for pts in self.countRioPts:
                idCounter += 1
                pts['intId'] = idCounter
                pts['Count'] = 0
                pts['CountG'] = 0
                pts['CountC'] = 0
                if pts['X'] == 1:
                    self.roiXmid += int(cv2.mean(np.array(pts['Points'])[:,0])[0])
                    self.roiXcount += 1
                if pts['Y'] == 1:
                    self.roiYmid += int(cv2.mean(np.array(pts['Points'])[:,1])[0])
                    self.roiYcount += 1
        pass
        if self.roiXcount != 0:
            self.roiXmid = int (self.roiXmid / self.roiXcount)
        if self.roiYcount != 0:
            self.roiYmid = int (self.roiYmid / self.roiYcount)
        

    def multiClassNMS(self, logname, boxes, confidences, classIDs, vis=False):
        # Bazı durumlarda model, kamyon yada otobüslere aynı araç için 
        # 2 farklı etiket ataması yapabiliyor. Bunu engellemek için farklı sınıflar arası NMS uygulandı.
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

        if False:    ## bboxlar kutu çizdirme kapatıldı.
            for bb in range(len(trueBoxes)):

                text = '{}: {:.4f}'.format(trueClassIds[bb], trueConf[bb])
                x, y = trueBoxes[bb][0], trueBoxes[bb][1]
                w, h = trueBoxes[bb][0] + trueBoxes[bb][2], trueBoxes[bb][1] + trueBoxes[bb][3] 
                cv2.rectangle(self.copyFrame,(x, y),(w, h), self.colors[trueClassIds[bb]], 2)       ##bboxlar çizdirilmesi
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
            self.vidW  = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.vidH = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
        except:
            print("[INFO] could not determine # of frames in video")
            print("[INFO] no approx. completion time can be provided")
            total = -1

    def createVideoWriter(self, fileName, fps=25.0, dims=(2592,1520)):
        # Tespit sonuclarini kayıt etmek için yeni bir output videosu oluşturmaktadır.
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        vid = cv2.VideoWriter(fileName, fourcc, fps, dims)
        return vid 

    def writeVideo(self,vid, frame):
        # Elde edilen sonuç framelerini oluşturulan çıkış videosuna yazar
        vid.write(frame)

    def releaseVideo(self):
        # Program sonlandığında oluşturulan videoyu kapatır.
        self.vidHeat.release()


## Hareketli araclarin heatmaplarinin oluşturulması
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
        for pts in countPts:
            text = '{}'.format(pts['Count'])
            ptsx, ptsy = pts['Points'][0]
            cv2.putText(self.copyFrameCountCombined, text, (ptsx, ptsy-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255),3)


## Araclarin belirli yerlerden gecis sayimi
    def plotRoisOnImage(self,im, countPts):
        # araçların üzerinden geçtiği zaman sayılacağı polygon alanları görüntü üzerinde çizdirir.
        copyIm = im.copy()
        for pts in countPts:
            text = '{}'.format(pts['Count'])
            ptsx, ptsy = pts['Points'][0]
            cv2.putText(copyIm, text, (ptsx, ptsy-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0),3)
            cv2.fillPoly(copyIm, np.int32([pts['Points']]), (0,255,0))
        self.copyFrameCount = cv2.addWeighted(copyIm, 0.45, im, 0.55, 0)
        
    def counter(self, logname, takip_edilen_arac):
        # Takip edilen noktalar herhangi bir sayım poligonuna düştüğü 
        # zaman ilgili poligonun sayım değerini artıran fonksiyon

        # inRois = False
        incounter = 0
        if len(logname) < 3:
            write_log = False
        else:
            write_log = True

        point = takip_edilen_arac.kutle_merkezleri[-1]
        idText = str(int(takip_edilen_arac.sinif)) + str(int(takip_edilen_arac.aracID)) 
        cv2.putText(self.copyFrame, idText, (point[0], point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,150,255), 2) # aracID araç üstüne yaz
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
            CenterX = self.vidW/2
            CenterY = self.vidH/2

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

                deltaX = km[kmCount-1][0] - km[kmCount-kmLim][0] ## pozitif sonuç = geçiş yönü sağ
                deltaY = km[kmCount-1][1] - km[kmCount-kmLim][1] ## pozitif sonuç = geçiş yönü aşağı

                girisX = False
                if (deltaX > 0 and CenterX > km[kmCount-1][0]) or (deltaX < 0 and CenterX < km[kmCount-1][0]):
                    ## sağa gidiyor ve kadrajda solda       veya       sola gidiyor ve kadrajda sağda
                    girisX = True              ## yatay eksene göre giriş yapmış
                
                girisY = False
                if (deltaY > 0 and CenterY > km[kmCount-1][1]) or (deltaY < 0 and CenterY < km[kmCount-1][1]):
                    ## aşağı gidiyor ve kadrajda yukarıda       veya       yukarı gidiyor ve kadrajda aşağıda
                    girisY = True              ## düşey eksene göre giriş yapmış
                

                directionX = False
                if abs(deltaX) > abs(deltaY):   ## Yatay eksendeki fark düşey eksendeki farktan daha büyükse
                    directionX = True           ## araç asıl yatay eksende hareket ediyor demektir

                #print("ROI ID=" + str(ptz['intId']) + ", x=" + str(ptz['X']) + ", y=" + str(ptz['Y']))
                #print("xdif = " + str(deltaX) + ",ydif = " + str(deltaY) + ",yatay = " + str(directionX) + ",Xgiris = " + str(girisX) + ",Ygiris = " + str(girisY))
                
                postt = False
                girisTuru = "Cikis"
                toplamRoi = 0

                if ptz['X'] == 1:                               ## ROI yatay eksen ile ilgili sayım yapıyor
                    if girisX:                                  ## aracın hareket yönü giriş mi
                        if takip_edilen_arac.ilkGecis == -1:    ## ilk geçiş mi
                            takip_edilen_arac.ilkGecis = ptz['intId']
                            ptz['Count'] += 1 
                            ptz['CountG'] += 1 
                            postt = True
                            girisTuru = "Giris"
                            toplamRoi = ptz['CountG']
                    else:                                       ## çıkış ise
                        if takip_edilen_arac.ikinciGecis == -1 and takip_edilen_arac.ilkGecis != ptz['intId']:   
                            takip_edilen_arac.ikinciGecis = ptz['intId']
                            ptz['Count'] += 1 
                            ptz['CountC'] += 1 
                            postt = True
                            toplamRoi = ptz['CountC']
                            if takip_edilen_arac.ilkGecis == -1:    ## çıkış yapan aracın girişi o araçla ilişkilendirilememiş
                                takip_edilen_arac.ilkGecis = 0

                elif ptz['Y'] == 1:                               ## ROI düşey eksen ile ilgili sayım yapıyor
                    if girisY:                                  ## aracın hareket yönü giriş mi
                        if takip_edilen_arac.ilkGecis == -1:    ## ilk giriş
                            takip_edilen_arac.ilkGecis = ptz['intId']
                            ptz['Count'] += 1
                            ptz['CountG'] += 1  
                            girisTuru = "Giris"
                            toplamRoi = ptz['CountG']
                            postt = True
                    else:
                        if takip_edilen_arac.ikinciGecis == -1 and takip_edilen_arac.ilkGecis != ptz['intId']:     ## çıkış yapmış
                            takip_edilen_arac.ikinciGecis = ptz['intId']
                            ptz['Count'] += 1 
                            ptz['CountC'] += 1 
                            toplamRoi = ptz['CountC']
                            postt = True
                            if takip_edilen_arac.ilkGecis == -1:    ## çıkış yapan aracın girişi o araçla ilişkilendirilememiş
                                takip_edilen_arac.ilkGecis = 0


                '''
                elif takip_edilen_arac.ilkGecis == -1:
                    takip_edilen_arac.ilkGecis = ptz['intId']
                    ptz['Count'] += 1   
                elif takip_edilen_arac.ikinciGecis == -1 and takip_edilen_arac.ilkGecis != ptz['intId']:
                    takip_edilen_arac.aracSayildiRoi = ptz['intId']
                    takip_edilen_arac.ikinciGecis = ptz['intId']
                    ptz['Count'] += 1 
                '''

                gonderilecekTarih = self.baslangicTarihi + timedelta(seconds=PTZTracking.startFrame/30)
                gonderilecekTarihs = gonderilecekTarih.strftime("%Y-%m-%dT%H:%M:%S+03:00")

                payload = json.dumps({
                "id": self.vidid,
                "Tur": girisTuru,
                "Loop": str(ptz['Id']),
                "Adet": toplamRoi
                })

                
                if postt and self.httppost:
                    PTZTracking.sendPayload(self, "Total", payload, postt)
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
    def trackDetections(self, frame, boxes, classIDs, roiPts, confidences):
        dikdortgen = []
        takipEdiciler = []
        dets = []

        if len(boxes) > 0:  ## TODO hic yoksa??
            i = 0
            for dd in confidences:
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                dets.append([x, y, x+w, y+h, dd, classIDs[i]])
                i += 1
                
        dets = np.asarray(dets)
        if len(dets) < 1:
            dets = np.empty((0, 6))
        tracks = self.mot_tracker.update(dets)
        previous = self.mot_memory
        self.mot_memory = {}
        
        Sort_boxes = []
        Sort_indexIDs = []
        Sort_c = []

        for track in tracks:
            Sort_boxes.append([track[0], track[1], track[2], track[3]])
            Sort_indexIDs.append(int(track[4]))
            Sort_c.append(int(track[5]))
            self.mot_memory[Sort_indexIDs[-1]] = Sort_boxes[-1]

        if len(tracks) > 0:
            i = 0
            notNull = True
            for box in Sort_boxes:
                (x, y) = (int(box[0]), int(box[1]))
                (w, h) = (int(box[2]), int(box[3]))
                color = (50,0,200)
                text = "id=" + str(Sort_indexIDs[i]) + ",cls=" + str(Sort_c[i])
                i += 1
                cv2.rectangle(self.copyFrame, (x, y), (w, h), color, 3)
                cv2.putText(self.copyFrame, text, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 4)
        else:
            notNull = False
            self.mot_memory = previous
        
        return tracks, notNull

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
            takipEdiciler.append([dlibKorelasyonTakipEdici, classID])       ## tracker
            notNull = True
        if len(takipEdiciler) == 0:
            dlibKorelasyonTakipEdici = dlibpast                     ## tracker - classes
            notNull = False
        return takipEdiciler, dlibKorelasyonTakipEdici, notNull       
    
    #def updateTrackPoss(self, frame, takipEdiciler, dlibKorelasyonTakipEdici):
    def updateTrackPoss(self, frame, tracks):
        Sort_dikdortgenler = []
        for i in tracks:
            baslangicX = int(i[0])
            baslangicY = int(i[1])
            bitisX = int(i[2])
            bitisY = int(i[3])
            Sort_dikdortgenler.append([(baslangicX, baslangicY, bitisX, bitisY), i[5]]) 
        return Sort_dikdortgenler

    def processTracks(self, logname, araclar, takipEdilenAraclar, aracSiniflari):
        # Takip edilen noktaların yeni gelen noktalarını ilgili nesneye ekleyerek sayım ve heatmap'leri çıkartan fonksiyon.
        
        for (aracID, kutle_merkezi) in araclar.items():
            takip_edilen_arac = takipEdilenAraclar.get(aracID, None)
            arac_sinif = aracSiniflari[aracID]
            
            if takip_edilen_arac is None:
                takip_edilen_arac = TakipEdilenAracClass(aracID, kutle_merkezi, arac_sinif)
            else:   
                takip_edilen_arac.kutle_merkezleri.append(kutle_merkezi) 
            takipEdilenAraclar[aracID] = takip_edilen_arac
            
            self.counter(logname, takip_edilen_arac) # Arac Sayım Fonksiyonu   
        self.plotRoisOnImage(self.copyFrame, self.countRioPts)

        # takipEdilenAraclar listesinde oluşturulmuş ve görüntüden çıkıp artık takibi olmayan nesneleri temizleyen kısım.     
        keys2Remove = []
        for key in takipEdilenAraclar:
            if key in araclar.keys(): continue
            keys2Remove.append(key)
            self.plotHeatmapLine(takipEdilenAraclar[key].kutle_merkezleri)
        self.plotHeatmapandCounts() ## self.plotHeatmap()   ### kapatıldı, eski hale getirmek için açılabilir

        for key in keys2Remove:     ## silinen araclar burada siliniyor ANALIZ BURADA KAVSAK 2022

            if len(logname) < 3:
                write_log = False
            else:
                write_log = True

            if write_log and takipEdilenAraclar[key].ilkGecis != -1:
                distlogname = logname[:len(logname)-4] + "_gecisler.txt"
                tempText = "aracID = " + str(takipEdilenAraclar[key].sinif) + str(takipEdilenAraclar[key].aracID) + " ilk = " + str(takipEdilenAraclar[key].ilkGecis) + " ikinci = " + str(takipEdilenAraclar[key].ikinciGecis) + ", frame =  " + str(PTZTracking.startFrame) + "\n"
                with open(distlogname, "a") as logtxt:
                    logtxt.write(tempText)
            
            if takipEdilenAraclar[key].ikinciGecis != -1:
                if write_log:
                    distlogname = logname[:len(logname)-4] + "_gecisler_kusursuz.txt"
                    tempText = "aracID = " + str(takipEdilenAraclar[key].sinif) + str(takipEdilenAraclar[key].aracID) + " ilk = " + str(takipEdilenAraclar[key].ilkGecis) + " ikinci = " + str(takipEdilenAraclar[key].ikinciGecis) +  ", frame =  " + str(PTZTracking.startFrame) + "\n"
                    with open(distlogname, "a") as logtxt:
                        logtxt.write(tempText)
                print("ilk = " + str(takipEdilenAraclar[key].ilkGecis))
                print("ikinci = " + str(takipEdilenAraclar[key].ikinciGecis))
                self.kavsakAnalizSayimlar[takipEdilenAraclar[key].ilkGecis][takipEdilenAraclar[key].ikinciGecis] += 1
                print(self.kavsakAnalizSayimlar)

                if write_log:
                    distlogname = logname[:len(logname)-4] + "_gecisler_kusursuz_array.txt"
                    with open(distlogname, "w") as logtxt:
                        logtxt.write("satirlar giris, sutunlar cikis" + "\n")
                        logtxt.write("ilk satir ve sutunlar girisleri belirtiyorlar" + "\n" + "\n")
                        for line in self.kavsakAnalizSayimlar:
                            logtxt.write(" ".join(str(line)) + "\n")
                    
                
                payload = json.dumps({
                "id": self.vidid,
                "G_Loop": str(takipEdilenAraclar[key].ilkGecis),
                "C_Loop": str(takipEdilenAraclar[key].ikinciGecis)
                })

                if self.httppost and takipEdilenAraclar[key].ilkGecis != 0:
                    PTZTracking.sendPayload(self, "Root", payload, True)
                
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
        # Video üzerinde her bir frame'i modele sokup gerekli takip ve sayım fonksiyonlarını çağıran ana döngü.
        self.heatMap = np.zeros((self.H, self.W), np.uint16)
        fr = 0
        self.httppost = httppostArg
        onetime=False

        exp = get_exp(None, args.xname)
        fasf = args.httppost

        device = "gpu"
        exp.test_conf = args.confidence
        exp.nmsthre = args.nmsth
        exp.test_size = (args.tsize,args.tsize)

        model = exp.get_model()

        model.cuda()
        model.eval()

        ckpt_file = args.weights
        ckpt = torch.load(ckpt_file, map_location="cpu") ##?
        model.load_state_dict(ckpt["model"])

        fp16 = False
        legacy = False

        predictor = Predictor(
        model, exp, COCO_CLASSES, None,
        device, fp16, legacy,
        )

        #vid_writer = self.createVideoWriter(str(self.outputfolder + args.output + '_YOLOX.avi'), fps=30.0, dims=(self.vidW,self.vidH))

        while True:

            grabbed, self.frame = self.vid.read()
            if not grabbed: break
            t1 = time.time()


            if onetime and self.httppost:

                gonderilecekTarih = self.baslangicTarihi + timedelta(seconds=PTZTracking.startFrame/30)
                if gonderilecekTarih.second == 0:
                    gonderilecekTarih + timedelta(11)
                gonderilecekTarihs = gonderilecekTarih.strftime("%Y-%m-%dT%H:%M:%S+03:00")

                url = "http://10.210.210.95:2101/WebApi/api/InfluxAdd_Total"
                payload = json.dumps({
                "id": self.vidid,
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
            framTxt = str(fr) + " " + args.xname[len(args.xname)-3:].upper() #model sol ust yazı text
            cv2.putText(self.frame, framTxt, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,250,255), 2) # framecounter

            PTZTracking.startFrame = fr
            #if fr%10 == 0:
            #    print('Frame -- {}'.format(fr))
            self.copyFrame = self.frame.copy()
            self.heatMapFrame = np.zeros((self.H, self.W), np.uint16)



            boxes = []
            classIDs = []
            confidences = []

            outputs, img_info = predictor.inference(self.frame)

            if outputs[0] is not None:
                output = outputs[0].cpu()
                cls = output[:, 6]
                scores = output[:, 4] * output[:, 5]
                xx2 = (outputs[0].cpu())[:,0:4] / img_info["ratio"] 
                a3 = xx2.numpy()
                a3[:,2] = a3[:,2] - a3 [:,0]
                a3[:,3] = a3[:,3] - a3 [:,1]

                boxes = a3
                classIDs = cls
                confidences = scores
            

            boxes, confidences, classIDs = self.multiClassNMS(logname, boxes, confidences, classIDs, vis)

            tracker, notNull = self.trackDetections(self.frame, boxes, classIDs, self.roiPts, confidences) 
            if notNull:
                dikdortgenler = self.updateTrackPoss(self.frame, tracker)
                araclar, aracSiniflari = self.kmt.guncelle(dikdortgenler)
                self.coolHeatmap(fr)
                self.takipEdilenAraclar = self.processTracks(logname, araclar, self.takipEdilenAraclar, aracSiniflari)
            
            ## Sayım videosunu kaydeder.
            #self.writeVideo(self.vidOut, self.copyFrameCount)
            ## Heatmap videosunu kaydeder.
            #self.writeVideo(self.vidHeat, self.copyFrameHeatMap)
            
            self.writeVideo(self.vidHeat, self.copyFrameCountCombined)

            if PTZTracking.startFrame % 10 == 0:
                print("frame = " + str(fr) + ", t = " + str(time.time()-t1) + ", FPS = " + str(1/(time.time()-t1))  )

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
    if args.output == "camoutDefault.avi":
        args.output = "out_" + str(PTZTracking.vidid)
    ptzTracker.vidCapture(args.input)
    #ptzTracker.vidOut = ptzTracker.createVideoWriter(str(PTZTracking.outputfolder + args.output + '.avi'), fps=30.0, dims=(PTZTracking.vidW,PTZTracking.vidH))
    ptzTracker.vidHeat = ptzTracker.createVideoWriter(str(PTZTracking.outputfolder + args.output + 'heatMap.avi'), fps=30.0, dims=(PTZTracking.vidW,PTZTracking.vidH))
    logname = PTZTracking.outputfolder + "log_" + args.output + ".txt"
    if True: ##TODO log yazmayı arg al
        logname = "no"
    ptzTracker.readRoiFromFile('selectedRoi.json')
    ptzTracker.readCountRoisFromFile('countRois.json')
    ptzTracker.runFrames(logname, ptzTracker.str2bool(args.httppost), vis=True)


if __name__ == '__main__':

    args = parseArgs()
    main(args)




## output folder = /home/acun/Desktop/PTZCamera/outputs/