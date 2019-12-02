import sys, os
import argparse
import numpy as np
import cv2 as cv
import json

sys.path.insert(0,'src/modules')
import face_recognizer as fr 
import book_recognizer as br

sys.path.insert(0, 'src/infrastructure')
from DynamicDatabase import *

uknownID = -1
count = 0

def createArgparse():
    parser = argparse.ArgumentParser(description='Smart Library Sample')
    parser.add_argument('-reid', type = str,#default = 'DNNfr',
                        dest = 'rdDet', required=True, help = 'Required. Type of recognizer. Available DNN face recognizer - DNNfr')
    parser.add_argument('-m_rd', type = str, #default = 'face-reidentification-retail-0095.xml',
                        dest = 'rdModel', required=True,  help = 'Required. Path to .xml file')        
    parser.add_argument('-fd', type = str, #default = 'DNNfd',
                        dest = 'fdDet', required=True, help = 'Required. Type of detector. Available DNN face detector - DNNfd')          
    parser.add_argument('-m_fd', type = str, #default = 'face-detection-retail-0004.xml',
                        dest = 'fdModel', required=True, help = 'Required. Path to .xml file')
    parser.add_argument('-lm', type = str, #default = 'DNNlm',
                        dest = 'lmDet',required=True,  help = 'Required. Type of detector. Available DNN landmarks regression - DNNlm')
    parser.add_argument('-m_lm', type = str, #default = 'landmarks-regression-retail-0009.xml',
                        dest = 'lmModel', required=True, help = 'Required. Path to .xml file')    

    parser.add_argument('-w_rd', type = int, default = '128',
                        dest = 'rdWidth', help = 'Optional. Image width to resize')
    parser.add_argument('-h_rd', type = int, default = '128',
                        dest = 'rdHeight', help = 'Optional. Image height to resize' ) 
    parser.add_argument('-t_rd', type = float, default = '0.8',
                        dest = 'rdThreshold', help = 'Optional. Probability threshold for face detections.' ) 

    parser.add_argument('-w_fd', type = int, default = '300',
                        dest = 'fdWidth', help = 'Optional. Image width to resize')
    parser.add_argument('-h_fd', type = int, default = '300',
                        dest = 'fdHeight', help = 'Optional. Image height to resize' ) 
    parser.add_argument('-t_fd', type = float, default = '0.9',
                        dest = 'fdThreshold', help = 'Optional. Probability threshold for face detections.' ) 

    parser.add_argument('-w_lm', type = int, default = '48',
                        dest = 'lmWidth', help = 'Optional. Image width to resize')
    parser.add_argument('-h_lm', type = int, default = '48',
                        dest = 'lmHeight', help = 'Optional. Image height to resize' ) 

    parser.add_argument('-br', type = str,  default='QR',
                        dest = 'br', help = 'Optional. Type - QR' )

    parser.add_argument('-lib', type = str, default='library.json',
                        dest = 'lib', help = 'Optional. Path to library.' ) 

    parser.add_argument('-w', type = int, default='0',
                        dest = 'web', help = 'Optional. Specify index of web-camera to open. Default is 0')
    args = parser.parse_args()
    return args

def createLibrary(libPath):
    with open(libPath, 'r', encoding='utf-8') as lib:
        data = json.load(lib)
    for book in data['books']:
        DB.addBook(book['id'], book['title'],  book['author'],
                   book['publisher'], book['year'])

def putText(img, text, pos, ix, iy, font, color, scale, thickness, rect = 1):

    textSize = cv.getTextSize(text, font, scale, thickness) 
    if rect:
        cv.rectangle(img, pos, (textSize[0][0] + ix, 
                pos[1]-textSize[0][1] + iy), (255, 255, 255), cv.FILLED)
    cv.putText(img, text, (pos[0], pos[1]+iy),
                font, scale, color, thickness)

def putInfo(img):
    indent = 10
    text = 'Show book'
    putText(img, text, (5,  img.shape[0]), 5, -5, cv.FONT_HERSHEY_SIMPLEX, 
            (22, 163, 245), 1, 2)

    text = 'Press:'
    putText(img, text, (5,  indent), 0, 0, cv.FONT_HERSHEY_PLAIN, 
            (22, 163, 245), 1, 1, 0)
    txtSize = cv.getTextSize(text, cv.FONT_HERSHEY_PLAIN, 1, 2) 

    indent += txtSize[0][1] + 5
    text = 'r - register'
    putText(img, text, (5,  indent), 0, 0, cv.FONT_HERSHEY_PLAIN, 
                        (22, 163, 245), 1, 1, 0)
    txtSize = cv.getTextSize(text, cv.FONT_HERSHEY_PLAIN , 1, 1) 

    indent += txtSize[0][1] + 5
    text = 'b - to get or ret a book'
    txtSize = cv.getTextSize(text, cv.FONT_HERSHEY_PLAIN , 1, 1) 
    putText(img, text, (5,  indent), 0, 0, cv.FONT_HERSHEY_PLAIN, 
            (22, 163, 245), 1, 1, 0)

    indent += txtSize[0][1] + 5
    text = 'f - get info' 
    txtSize = cv.getTextSize(text, cv.FONT_HERSHEY_PLAIN , 1, 1)
    putText(img, text, (5,  indent), 0, 0, cv.FONT_HERSHEY_PLAIN, 
            (22, 163, 245), 1, 1, 0)

def recUser(img):
    faces, out = faceRec.recognize(img)
    userID = uknownID
    for face in faces:
        if len(faces) > 1:
            text = 'No more than one person at a time'
            putText(img, text, (0,30), 0, -5, cv.FONT_HERSHEY_SIMPLEX, 
                                           (22, 163, 245), 1, 2)

        if np.amax(out) > faceRec.threshold:
            userID = int(np.argmax(out) + 1)
            text = 'User #' + str(userID)
            putText(img, text, face[0], face[0][0], -5, cv.FONT_HERSHEY_SIMPLEX, 
                                           (22, 163, 245), 1, 2)
        else:
            text = 'Unknown'
            putText(img, text, face[0], face[0][0], -5, cv.FONT_HERSHEY_SIMPLEX, 
                                           (22, 163, 245), 1, 2)     
        cv.rectangle(img, face[0], face[1], (22, 163, 245), 2)
    return userID

def printInfo(count):
    os.system('cls')
    if count == 0:
        DB.printUsers()
    elif count == 1:
        DB.printBooks()
    elif count == 2:
        DB.printBBooks()

def recBook(img):
    data = bookRec.recognize(img)
    try: 
        bID = int(data.split(' ')[0])
        print(bID)
        return bID 
    except ValueError:
        return -1
def main():
    args = createArgparse()
    brArgs = dict(name='')
    rdArgs = dict(name = '', rdXML = '', rdWidth= 0, rdHeight= 0, rdThreshold= 0,
    fdName = '', fdXML = '', fdWidth = 0, fdThreshold= 0,
    lmName = '', lmXML= 0, lmWidth= 0, lmHeight= 0)
    DB = DynamicDB()

    if (args.rdDet != None and args.fdDet != None and args.lmDet != None):
        rdArgs ['name'] = args.rdDet
        print(args)
        if (args.rdModel != None):
            rdArgs ['rdXML'] = args.rdModel
        if (args.rdWidth != None):
            rdArgs ['rdWidth'] = args.rdWidth
        if (args.rdHeight != None):
            rdArgs ['rdHeight'] = args.rdHeight
        if (args.rdThreshold != None):
            rdArgs ['rdThreshold'] = args.rdThreshold

        rdArgs ['fdName'] = args.fdDet
        if (args.fdModel != None):
            rdArgs ['fdXML'] = args.fdModel
        if (args.fdWidth != None):
            rdArgs ['fdWidth'] = args.fdWidth
        if (args.fdHeight != None):
            rdArgs ['fdHeight'] = args.fdHeight
        if (args.fdThreshold != None):
            rdArgs ['fdThreshold'] = args.fdThreshold

        rdArgs ['lmName'] = args.lmDet
        if (args.lmModel != None):
            rdArgs ['lmXML'] = args.lmModel
        if (args.lmWidth != None):
            rdArgs ['lmWidth'] = args.lmWidth
        if (args.lmWidth != None):
            rdArgs ['lmHeight'] = args.lmHeight
        if (args.br != None): 
            brArgs['name'] = args.br
        if (args.br != None): 
            lib = args.lib
        if (args.web != None): 
            src = args.web
        
        createLibrary(lib)
        
        bookRec = br.BookRecognizer.create(brArgs)
        faceRec = fr.FaceRecognizer.create(rdArgs)
        cap = cv.VideoCapture(src)
        identified = False
        while(True):
            _, img = cap.read()
            
            ch = cv.waitKey(5) & 0xFF

            userID = recUser(img)   
            
            if userID != uknownID:
                putInfo(img)
                if ch == ord('b'):
                    bookID = recBook(img)
                    os.system('cls')
                    print(bookID)
                    DB.getRetBook(userID, bookID)
                    DB.printBBooks()
                
            elif ch  == ord('r'):
                n = faceRec.register(img)
                DB.addUser(n)
                text = 'You are user #' +  str(n)
                putText(img, text, (5,  25), 5, -5, cv.FONT_HERSHEY_SIMPLEX, 
                                            (22, 163, 245), 1, 2)
                os.system('cls')
                DB.printUsers()
                cv.imshow('window',  img)
                cv.waitKey(1000)
            
            cv.imshow('window',  img)
            if ch == ord('f'):
                count = count + 1 
                printInfo(count % 3)

            if ch == ord('q'):
                break

        cap.release()
        
if __name__ == '__main__':
    sys.exit(main() or 0)
