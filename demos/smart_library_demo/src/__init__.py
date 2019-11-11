import sys, os, re
import numpy as np
import argparse 
import cv2 as cv
from datetime import datetime, date, time

sys.path.insert(0, 'modules')
import face_recognizer
import book_recognizer

sys.path.insert(0, "infrastructure")
from CSVDatabase import *
from Entities.User import *
from Entities.Book import *

CSV = CSVDatabase()
brName = "ORB"

def build_argparse():
    parser = argparse.ArgumentParser(description='OpenVINO Smart Library')
    parser.add_argument('-r', '--recognizer', type = str, default = 'PVL',
                        dest = 'recognizer', help = 'type of recognizer')
    parser.add_argument('-d', '--dll', type = str, default = '../samples/PVL_wrapper.dll',
                        dest = 'dll', help = 'dll')
    parser.add_argument('-m', '--model', type = str, default = 'defaultdb.xml' ,
                        dest = 'model', help = 'data base with faces')
    args = parser.parse_args()
    return args

def showText(f, x, y, h, w, name):
     cv.rectangle(f, (x, y), (x + w, y + h), (0, 255, 0), 1)
     cv.putText(f, name , (x,y-2), cv.FONT_HERSHEY_SIMPLEX, 
                    1, (219, 132, 58), 2)
     cv.putText(f, "Press Q to exit" , (10,40), 
                          cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 208, 86), 1)

    #  FPS_MEASURE_INTERVAL = 30
    #  fpsMeasure.fpsSum += elapsed
    #  fpsMeasure.fpsInterval = fpsMeasure.fpsInterval + 1
    #  if (fpsMeasure.fpsInterval == FPS_MEASURE_INTERVAL):
    #       fpsMeasure.fps = 1.0 / fpsMeasure.fpsSum * FPS_MEASURE_INTERVAL
    #       fpsMeasure.fpsInterval = 0
    #       fpsMeasure.fpsSum = 0
     
    #  string = "fps:" + "{0:.3f}".format(fpsMeasure.fps)
    #  cv.putText(f, string,(10,60), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 208, 86), 1)

def signUp(ID):
  CSV = CSVDatabase()
  print("Please input your personal information here")
  print("First name:")
  fName = input() # first name 
  print("Last name:")
  lName = input() # last name
  print("Middle name:")
  mName =  input() # middle name
  print("phone:")
  phone =  input()
  user = User(ID, phone, fName, lName, mName)
  user._print()
  newID = CSV.AddUser(user)
  print("Result: ")
  if(newID == -1):
    print("This user is already registered")
  else:
    print("Done!")

def getCovers():
        templ = []
        desTpl = []
        for i in os.listdir("infrastructure/Database/Books/Covers/"):
           if (re.fullmatch('([^\s]+(\.(?i)(jpg|png|gif|bmp))$)', i)):
                templ.append(os.path.join("infrastructure/Database/Books/Covers/", i))
        #Список с ключевыми точками шаблонов
        det = cv.ORB_create()
        for t in templ:
            tpl = cv.imread(t)
            tplGray = cv.cvtColor(tpl, cv.COLOR_BGR2GRAY)
            _, tmp = det.detectAndCompute(tplGray, None)
            desTpl.append(tmp)

        return [len(templ), desTpl]
def addBook():
  """"""  
def getOrRetBook(frame): 
  rec = book_recognizer.Recognizer()
  rec.create(brName)

  resArr = []
  desTpl = []
  l, desTpl = getCovers()
  print(l)
  _, frame = cap.read()
  ym, xm, _ = frame.shape
  i = 0
  for i in range(l):
      resArr.append(0)


  print("book recognition")
  print(resArr)
  cropFrame = frame[ym//2 - 170 : ym//2 + 170,
                        xm//2 - 120 : xm//2 + 120]
  cv.rectangle(frame, (xm//2 - 110, ym//2 - 150), 
                    (xm//2 + 110, ym//2 + 145), (0, 255, 255))    
  numF = len([f for f in os.listdir('infrastructure/Database/Books/Covers/')
      if os.path.isfile(os.path.join('infrastructure/Database/Books/Covers/', f))]) - 1
      
  if (numF != l):
      l, desTpl = getCovers()
      resArr.clear()
      for i in range(l):
          resArr.append(0)
          
  if (resArr):
      recognizeResult = rec.recognize(cropFrame, desTpl, 0.7)
      out = str(100 * max(resArr) / 400)
      out = out + '%'
      cv.putText(frame, out, (200, 200), cv.FONT_HERSHEY_SIMPLEX,
                                                1, (0, 255, 255), 2)
      for i in range(l):
          resArr[i] = resArr[i] + recognizeResult[i]
      if max(resArr) > 200:
          ID = resArr.index(max(resArr))+1
          print(ID)
          resArr.clear()
          for i in range(l):
              resArr.append(0)
  return frame
                    

recArgs = dict(name = '', dll = '' , db= '')
args = build_argparse()

if (args.recognizer != None):
  recArgs['name'] = args.recognizer
  if (args.dll != None):
    recArgs['dll'] = args.dll
  if (args.model != None):
    recArgs['db'] = args.model
  rec = face_recognizer.FaceRecognizer.create(recArgs)
  uID = rec.getUID()


ch = ' '
hold = 1  
cap = cv.VideoCapture(0)
ID = uID 
entered = False
while(True): 
  _, f = cap.read()
  
  if ID == uID and not entered:
    (ID, (x, y, w, h)) = rec.recognize(f)
  elif ID != uID and not entered :
    (_ , (x, y, w, h)) = rec.recognize(f)

  if not entered:
    if ID != uID:
      name = (CSV.GetUser(ID))[0].first_name
      cv.putText(f, "Press Enter to Sign In" , (10,460), 
                            cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 208, 86), 1)
    else:
      name = "UNKNOWN"
      cv.putText(f, "You are not a member" , (10,460), 
                            cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 208, 86), 1)
      cv.putText(f, "Press R to register." , (10,20), 
                            cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 208, 86), 1)
    showText(f,x,y,h,w,name)
    if ch & 0xFF == ord('r') and ID == uID:
      ID = rec.getNewID()
      tmp = rec.register(f, ID)
      signUp(ID)
    if ch  & 0xFF == ord(' ') :
      entered = True
  else:
    cv.putText(f, "Place book in the selected area" , (10,460), 
                            cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 208, 86), 1)
    if ch  & 0xFF == ord('r') :
      getOrRetBook(frame)

  cv.imshow("OpenVINO Smart Library", f)
  ch = cv.waitKey(hold)

  if ch & 0xFF == ord('q'):
    break

cap.release()