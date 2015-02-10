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

###################################################################

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

###################################################################
#initialize variables/define constants 

plt.ion() # set plot to interactive mode (for life time)

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

cnt = 0

timecount = 1

arduinoData = serial.Serial(comPort,115200)

time = []
accX = []
accY = []
accZ = []
gyrX = []
gyrY = []
gyrZ = []
plot =[]
filteredX = []
filteredY = []

#####################################################################
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
#####################################################################

# make the real time figure
def makeFig():
    plt.plot(filteredX,'r-')

#####################################################################

def initializeCompFilter(time,accX,accY,accZ,gyrX,gyrY,gyrZ):
    arrayLength = len(time)

    accXavg = (sum(accX)/arrayLength) - ACCEL_ZERO_G
    accYavg = (sum(accY)/arrayLength) - ACCEL_ZERO_G
    accZavg= (sum(accZ)/arrayLength) - ACCEL_ZERO_G
    gyrXavg = sum(gyrX)/arrayLength
    gyrYavg = sum(gyrY)/arrayLength
    gyrZavg = sum(gyrZ)/arrayLength

    one_gee = math.sqrt(accXavg**2 + accYavg**2 + accZavg**2)

    my_x_comp_filter = ComplementaryFilter(SamplePeriod = T,BandWidth = k,
                    Gyro_zero = 0., On_Axis_zero = accYavg,Off_Axis_Zero=accXavg,
                    Z_Axis_zero=accZavg,One_Gee=one_gee)
 
 ##for some reason, the on and off axis for this are both "z"
    my_y_comp_filter = ComplementaryFilter(SamplePeriod = T,BandWidth = k,
                    Gyro_zero = 0., On_Axis_zero = accXavg,Off_Axis_Zero=accZavg,
                    Z_Axis_zero=accZavg,One_Gee=one_gee)

    return my_x_comp_filter, my_y_comp_filter, accXavg, accYavg, accZavg, gyrXavg, gyrYavg, gyrZavg


#######################################################################
def compFilter(my_x_comp_filter, my_y_comp_filter,time,accX,accY,accZ,gyrX,gyrY,gyrZ,gyrXavg, gyrYavg, gyrZavg):

        
    for i in range(len(time)):
        xgyrodeg = (gyrX[i]-gyrXavg)/1.024 * AREF / 2. 
        ygyrodeg = (gyrY[i]-gyrYavg)/1.024 * AREF / 2. 
        zgyrodeg = (gyrZ[i]-gyrZavg)/1.024 * AREF / 3.3
        xacc = accX[i] - ACCEL_ZERO_G
        yacc = accY[i] - ACCEL_ZERO_G
        zacc = accZ[i] - ACCEL_ZERO_G

        x_comp_filter = my_x_comp_filter(Gyro_input=math.radians(xgyrodeg), On_Axis_input=yacc, Off_Axis_input=xacc,Z_Axis_input=zacc)
        y_comp_filter = my_y_comp_filter(Gyro_input=math.radians(ygyrodeg), On_Axis_input=xacc, Off_Axis_input=yacc,Z_Axis_input=zacc)
        filteredX.append(x_comp_filter)
        filteredY.append(y_comp_filter)

    return filteredX, filteredY



#######################################################################
#Main Functions
try:
    ##grab the data
    while True:
        for line in arduinoData:
            if (cnt == 0):
                print 'Processing...'
            arduinoString = arduinoData.readline()
            dataArray = arduinoString.split(',')
            if(cnt>5):
                if(cnt == 6):
                    print 'Starting Data Acquision...'
                t = float(dataArray[0])
                aX = float(dataArray[1])
                aY = float(dataArray[2])
                aZ = float(dataArray[3])
                gX = float(dataArray[4])
                gY = float(dataArray[5])
                gZ = float(dataArray[6])
                time.append(t)
                accX.append(aX)
                accY.append(aY)
                accZ.append(aZ)
                gyrX.append(gX)
                gyrY.append(gY)
                gyrZ.append(gZ)
                print t

                ##filtering 
                if(t>3000*timecount):
                    if (timecount == 1): 
                        my_x_comp_filter, my_y_comp_filter, accXavg, accYavg, accZavg, gyrXavg, gyrYavg, gyrZavg = initializeCompFilter(time,accX,accY,accZ,gyrX,gyrY,gyrZ) 
                    if (timecount > 1):
                        filteredX, filteredY = compFilter(my_x_comp_filter, my_y_comp_filter,time,accX,accY,accZ,gyrX,gyrY,gyrZ,gyrXavg, gyrYavg, gyrZavg)
                        drawnow(makeFig)
                        print filteredX
                        
                    timecount=timecount+1

            cnt = cnt + 1
except KeyboardInterrupt:
    pass

