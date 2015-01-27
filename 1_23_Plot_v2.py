#!/usr/bin/python
# Live Visualization

import serial
import numpy
import matplotlib.pyplot as plt
from drawnow import *

accX = []
accY = []
arduinoData = serial.Serial('/dev/cu.usbmodem1411',9600)
plt.ion()
cnt = 0

def makeFig():
    plt.plot(accX,'r-')

while (cnt<1000):
    while (arduinoData.inWaiting()==0):
        pass
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
        

        
