#!/usr/bin/python
# Live Visualization

import serial
import numpy
import matplotlib.pyplot as plt
from drawnow import *
import csv

accX = []
accY = []
arduinoData = serial.Serial('/dev/cu.usbmodem1411',9600)
plt.ion()
cnt = 0
datacnt = 0

def makeFig():
    plt.plot(accX,'r-')

try:
    while True:
        for line in arduinoData:
            print 'Hi'
            arduinoString = arduinoData.readline()
            dataArray = arduinoString.split(',')
            if(cnt>5):
                X = float(dataArray[0])
                Y = float(dataArray[1])
                accX.append(X)
                accY.append(Y)
                drawnow(makeFig)
                if(cnt>100):
                    accX.pop(0)
                    accY.pop(0)
            cnt = cnt+1
except KeyboardInterrupt:
    pass

with open('/Users/rushipatel/Documents/SeniorDesignApps/PythonScripts/saveFile5.csv', 'wb') as csvfile:
    while(datacnt<len(accX)):
        dataWriter = csv.writer(csvfile)
        dataWriter.writerow([accX[datacnt]]+[accY[datacnt]])
        datacnt = datacnt + 1
         
            

            
