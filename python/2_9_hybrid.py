# pythonPlot.py
#Team Aura, Rice University Senior Design 2014-2015

##################################################################

#import modules 
import serial
import string
import time
import math
import matplotlib.pyplot as plt
from drawnow import *
import csv
import sys
from numpy import NaN, Inf, arange, isscalar, asarray, array
import numpy




# comp filter bandwidth (rad/s)
k = 4. #was 0.25

#pushpak timeframe (sec)
T = 1/100.

# AVR ADC reference voltage (volts - from previous experiments)
#   should probably be computed on-board
AREF = 2.82

# zero g value for MMA7260Q accelerometer (ADC counts - from data sheet and AREF)
#   should probably be computed on-board
ACCEL_ZERO_G = 599.

# *****************************************************************
#File Saving and User Definitions 
file_name = raw_input('save file name without spaces!! : ')
developer = raw_input('are you rushi or hannelle? : ')

if (developer=="rushi" or developer=="Rushi" or developer=="r"):
    comPort = '/dev/cu.usbmodem1411'
    savepath = '/Users/rushipatel/Documents/SeniorDesignApps/PythonScripts/'
    
elif (developer=="hannelle" or developer=="Hannelle" or developer=="h"):
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
# *****************************************************************

#initialize variables 

time = []
accX = []
accY = []
accZ = []
gyrX = []
gyrY = []
gyrZ = []
plot =[]
plot_time = [0, 0, 0, 0, 0, 0, 0, 0, 0]

arduinoData = serial.Serial(comPort,115200)

plt.ion()
cnt = 0
datacnt = 0
timecount = 1

# define integrator class for use in comp filter(rectangular for now)
class Integrator:
    def __init__(self, SamplePeriod = 1., InitialCondition = 0.):
        self.T = SamplePeriod
        self.State = InitialCondition
    def __call__(self, Input=0.):
        self.State += self.T * Input
        return self.State
        
# define comp filter class
class ComplementaryFilter:
    def __init__(self,SamplePeriod,BandWidth,Gyro_zero,On_Axis_zero,Off_Axis_Zero,Z_Axis_zero,One_Gee):
        self.k = BandWidth
        self.one_gee = One_Gee
        self.Internal = Integrator(SamplePeriod,-Gyro_zero)
        self.Perpendicular = math.sqrt(Off_Axis_Zero**2 + Z_Axis_zero**2)
        self.Output = Integrator(SamplePeriod,-math.atan2(On_Axis_zero,self.Perpendicular))
        self.Prev_Output = self.Output()
        
    def __call__(self,Gyro_input,On_Axis_input,Off_Axis_input,Z_Axis_input):
        self.gmag = math.sqrt(On_Axis_input**2 + Off_Axis_input**2 + Z_Axis_input**2)
        self.Gyro_in = Gyro_input
        self.Perpendicular = math.sqrt(Off_Axis_input**2 + Z_Axis_input**2)
        self.angle = -math.atan2(On_Axis_input,self.Perpendicular)
        if  abs(self.gmag-self.one_gee)/self.one_gee >0.05:
            self.input1 = 0.
        else:
            self.input1 = (self.angle - self.Prev_Output)
        self.temp = self.Internal(self.input1*self.k*self.k)    
        self.input2 = self.temp + (self.input1)*2*self.k - Gyro_input
        self.temp = self.Output(self.input2)
        self.Prev_Output = self.temp
        return self.temp


def makeFig():
    plt.plot(plot,'r-')

#def filter(time,accX,accY,accZ,gyrX,gyrY,gyrZ):
#	for i in range(len(time)):
#		one_gee = math.sqrt(accX**2 + accY**2 + accZ**2)
#		anglez = 0
#		my_x_comp_filter = ComplementaryFilter(SamplePeriod = T,BandWidth = k,
 #                   Gyro_zero = 0., On_Axis_zero = yacc,Off_Axis_Zero=xacc,
  #                  Z_Axis_zero=zacc,One_Gee=one_gee)


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
				
				if (cnt == 50):
					avgX = sum(accX)/cnt
					avgY = sum(accY)/cnt
					avgZ = sum(accZ)/cnt
					one_gee = math.sqrt(avgX**2 + avgY**2 + avgZ**2)

					my_x_comp_filter = ComplementaryFilter(SamplePeriod = T,BandWidth = k,
                    Gyro_zero = 0., On_Axis_zero = avgY,Off_Axis_Zero=avgX,
                    Z_Axis_zero=avgZ,One_Gee=one_gee)
					my_y_comp_filter = ComplementaryFilter(SamplePeriod = T,BandWidth = k,
                    Gyro_zero = 0., On_Axis_zero = avgX,Off_Axis_Zero=avgY,
                    Z_Axis_zero=avgZ,One_Gee=one_gee)

				if (cnt>50):
					x_comp_filter = my_x_comp_filter(Gyro_input=(gX), On_Axis_input=aY, Off_Axis_input=aX,Z_Axis_input=aZ)
					y_comp_filter = my_y_comp_filter(Gyro_input=(gY), On_Axis_input=aX, Off_Axis_input=aY,Z_Axis_input=aZ)
					plot.append(x_comp_filter)

				if(t2>60000*timecount):
					print timecount
					timecount=timecount+1
			 		filter(time,accX,accY,accZ,gyrX,gyrY,gyrZ)
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
