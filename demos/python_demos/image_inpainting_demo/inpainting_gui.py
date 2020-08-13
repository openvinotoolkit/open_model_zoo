import cv2
import numpy as np
from inpainting import ImageInpainting

from openvino.inference_engine import IECore

class InpaintingGUI(object):

    def __init__(self, imgPath, modelPath, device="CPU"):
        self.wndName="Inpainting (press H for help)"
        self.maskColor=(255,0,0)
        self.radius=10
        self.oldPoint=(-1,-1)

        self.inpainter = ImageInpainting(IECore(),modelPath,device)
        self.imgPath = imgPath

        cv2.namedWindow(self.wndName, cv2.WINDOW_AUTOSIZE )
        cv2.setMouseCallback(self.wndName,self.onMouse)
        cv2.createTrackbar("Brush size",self.wndName,self.radius,30,self.onTrackBar)
        cv2.setTrackbarMin("Brush size",self.wndName,1);

        self.isHelpShown=False
        self.isOriginalShown=False

    def onMouse(self, event, x,y, flags, param):
        if flags==cv2.EVENT_FLAG_LBUTTON and not self.isOriginalShown:
            if self.oldPoint[0]!=-1:
                cv2.line(self.mask,self.oldPoint,(x,y),1,self.radius*2)
            cv2.circle(self.mask,(x,y),self.radius,1,cv2.FILLED)
            self.oldPoint=(x,y)

            self.updateWindow()
        else:
            self.oldPoint=(-1,-1)

    def run(self):

        srcImg = cv2.imread(self.imgPath, cv2.IMREAD_COLOR)
        self.img = cv2.resize(srcImg,(self.inpainter.input_width,self.inpainter.input_height))
        self.originalImg = self.img.copy();
        self.label = ""
        self.mask = np.zeros((self.inpainter.input_height,self.inpainter.input_width,1),dtype=np.float32)
        self.updateWindow()

        if self.img is None:
            print("Cannot load source image\n");
            return

        key= cv2.waitKey(1)
        while key not in (27,ord('q'),ord('Q')):
            if key==ord(" "):
                self.isOriginalShown = False;
                self.showInfo("Processing...")

                self.img[:,:,0:1][self.mask>0]=0
                self.img[:,:,1:2][self.mask>0]=0
                self.img[:,:,2:3][self.mask>0]=0
                self.img=self.inpainter.process(self.img,self.mask)

                self.showInfo("")
                self.mask[:,:,:]=0
                self.updateWindow()
            elif key in (8,ord('c'),ord('C')): # Backspace or c
                self.isOriginalShown = False;
                self.mask[:,:,:]=0
                self.updateWindow()
            elif key == ord('\t'):
                self.isOriginalShown = not self.isOriginalShown
                self.updateWindow()
            elif key in (ord('h'),ord('H')):
                if not self.isHelpShown:
                    self.showInfo("Use mouse with LMB to paint\nBksp or C to clear\nSpace to inpaint\nTab to show original image\nEsc or Q to quit")
                    self.isHelpShown=True
                else:
                    self.showInfo("")
                    self.isHelpShown=False

            key = cv2.waitKey(1)

    def onTrackBar(self,x):
        self.radius=x

    def showInfo(self,text):
        self.label=text
        self.updateWindow()
        cv2.waitKey(1) # This is fequired to actually paint window contents rigth away
        self.isHelpShown=False # Any other label removes help from the screen

    def updateWindow(self):
        pad=10
        margin=10
        if self.isOriginalShown:
            backbuffer=self.originalImg.copy()
            sz = cv2.getTextSize("Original", cv2.FONT_HERSHEY_COMPLEX, 0.75,1)[0]
            imgWidth = backbuffer.shape[1]
            labelArea = backbuffer[margin:sz[1]+pad*2+margin,imgWidth-margin-(sz[0]+pad*2):imgWidth-margin]
            labelArea//=2
            cv2.putText(backbuffer,"Original",(imgWidth-margin-sz[0]-pad,margin+sz[1]+pad),cv2.FONT_HERSHEY_COMPLEX, 0.75,(128,255,128))
        else:
            backbuffer = self.img.copy()
            backbuffer[:,:,0:1][self.mask>0]=self.maskColor[0]
            backbuffer[:,:,1:2][self.mask>0]=self.maskColor[1]
            backbuffer[:,:,2:3][self.mask>0]=self.maskColor[2]

        if self.label is not None and self.label!="":
            lines = self.label.split("\n")
            count = len(lines)
            w = max(cv2.getTextSize(line, cv2.FONT_HERSHEY_COMPLEX, 0.75,1)[0][0] for line in lines)+pad*2
            lineH = cv2.getTextSize(lines[0], cv2.FONT_HERSHEY_COMPLEX, 0.75,1)[0][1]+pad;
            labelArea=backbuffer[margin:lineH*count+pad*2+margin,margin:w+margin]
            labelArea//=2
            for i,line in enumerate(lines):
                cv2.putText(backbuffer,line,(pad+margin,margin+(i+1)*lineH),cv2.FONT_HERSHEY_COMPLEX, 0.75,(192,192,192))

        cv2.imshow(self.wndName, backbuffer)

