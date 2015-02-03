#!/usr/bin/python
# Live Visualization

import serial
import numpy
import matplotlib.pyplot as plt
from drawnow import *
import csv
import time


# *****************************************************************
# below code is for determining where saving is done

file_name = raw_input('save file name without spaces!! : ')
developer = raw_input('are you rushi or hannelle? : ')

if (developer=="rushi" or developer=="Rushi"):
	comPort = '/dev/cu.usbmodem1411'
	savepath = '/Users/rushipatel/Documents/SeniorDesignApps/PythonScripts/'
elif (developer=="hannelle" or developer=="Hannelle"):
	comPort = 3
	savepath = 'C:/Users/Hannelle/Documents/0_SrDs_Data/'
else:
	import win32com.client
	#finds COM port that the Arduino is on (assumes only one Arduino is connected)
	wmi = win32com.client.GetObject("winmgmts:")
	for port in wmi.InstancesOf("Win32_SerialPort"):
		#print port.Name #port.DeviceID, port.Name
		if "Arduino" in port.Name:
			comPort = port.DeviceID
			print comPort, "is Arduino"
	savepath = 'C:/'

# end of setup for saving
# *****************************************************************
	
time = []
accX = []
accY = []
accZ = []
gyrX = []
gyrY = []
gyrZ = []
plot_time = [0, 0, 0, 0, 0, 0, 0, 0, 0]

arduinoData = serial.Serial(comPort,115200)

plt.ion()
cnt = 0
datacnt = 0
timecount = 1


def makeFig():
    plt.plot(accX,'r-')

try:
    while True:
        for line in arduinoData:
			if (cnt == 0):
				print 'Processing...'
			arduinoString = arduinoData.readline()
			dataArray = arduinoString.split(',')
			if(cnt>5):
				if(cnt == 6):
					print 'Starting Data Acquision...'
				t2 = float(dataArray[0])
				aX = float(dataArray[1])
				aY = float(dataArray[2])
				aZ = float(dataArray[3])
				gX = float(dataArray[4])
				gY = float(dataArray[5])
				gZ = float(dataArray[6])
				time.append(t2)
				accX.append(aX)
				accY.append(aY)
				accZ.append(aZ)
				gyrX.append(gX)
				gyrY.append(gY)
				gyrZ.append(gZ)
				#break every 60 seconds
				if(t2>60000*timecount):
					print timecount
					timecount=timecount+1
				
				drawnow(makeFig)
			cnt = cnt+1
except KeyboardInterrupt:
    pass

with open(savepath+file_name+'.csv', 'wb') as csvfile:
    print 'Saving Data...'
    while(datacnt<len(accX)):
        dataWriter = csv.writer(csvfile)
        dataWriter.writerow([elapsed_time[datacnt]]+[accX[datacnt]]+[accY[datacnt]]+[accZ[datacnt]]+[gyrX[datacnt]]+[gyrY[datacnt]]+[gyrZ[datacnt]])
        datacnt = datacnt + 1          
