# PushpakSensorCubeDemo.py
#
# First public release - Rev 1.0 (5 Feb 2010)
#
# Copyright Roy Brewer 2010
#
# Reads gyro and accel sensor data from Puskback board and uses it to drive two 
# spinning cubes, one just integrating gyro data, the other implementing linear 
# complimentary filters. Uses Vpython. Assumes data is raw (not calibrated) and 
# oversampled 15x
#
# Details on the Pushpak custom quadrotor controller board can be found here:
#    <http://sites.google.com/site/pushpakquadrotor/Home>
#
# This example python script is free software: you can redistribute it and/or 
# modify it under the terms of the GNU General Public License as published by the 
# Free Software Foundation, either version 2 of the License, or (at your option) 
# any later version.
# The example python script is distributed in the hope that it will be useful, but 
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or 
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
# You should have received a copy of the GNU General Public License. If not, see 
# <http://www.gnu.org/licenses/>.
#
# History
# Rev 1.0 (5 Feb 2010) - First public release
# Rev 1.1 (10 Feb 2010) - Removed "don't move" window; commented out debug file


##################################################################


import serial
import string
import time
import math
import matplotlib.pyplot as plt
from drawnow import *
import csv

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

plt.ion() # set plot to animated
# Define constants
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


def makeFig():
    plt.plot(compX,'r-')


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
        
# Define routines to processes Pushpak data. Alter this to suit your on-board 
# data formats. My Pushpak puts out 7 comma separated sets of ASCII numbers 
# (timeframe and 6 sensors). The sensor outputs are raw (uncalibrated)  
# 15x oversampled ADC values.
class Process_Pushpak_Data:
    previous_input = "0,0,0,0,0,0,0\n"
    
    def Read_Data(self):
        try:
            line = self.ser.readline()   
            input = string.split(line, ',')
            timeframe = string.atof(input[0])	
            xaccel = string.atof(input[1])/15.
            yaccel = string.atof(input[2])/15.
            zaccel = string.atof(input[3])/15.
            xgyro = string.atof(input[4])/15.
            ygyro = string.atof(input[5])/15.
            zgyro = string.atof(input[6])/15.
        except: # if anything goes wrong, re- use the previous input
            print("Error reading line, using previous data")
            line = self.previous_input  
            input = string.split(line, ',')
            timeframe = string.atof(input[0])	
            xaccel = string.atof(input[1])/15.
            yaccel = string.atof(input[2])/15.
            zaccel = string.atof(input[3])/15.
            xgyro = string.atof(input[4])/15.
            ygyro = string.atof(input[5])/15.
            zgyro = string.atof(input[6])/15.
        self.previous_input = line
        return xaccel, yaccel, zaccel, xgyro, ygyro, zgyro
        
    def __init__(self,Com_Port,Baud):
        #open serial port
        self.ser = serial.Serial(Com_Port ,Baud) #to open COM4, use value 3.
        self.ser.setTimeout(1) # time out after 1 second
        # clear data buffer to prevent spurious data when XBee is first plugged into PC
        self.ser.flushInput()
        
        xaccavg = 0.
        yaccavg = 0.
        zaccavg = 0.
        self.xgyrozero = 0.
        self.ygyrozero = 0.
        self.zgyrozero = 0.

        
        # determine gyro zero values, sample time, and the accel sensor representation of 1 g
        time1 = time.clock()
        for i in range(100):
            xaccel, yaccel, zaccel, xgyro, ygyro, zgyro = self.Read_Data()
            xaccavg += xaccel
            yaccavg += yaccel
            zaccavg += zaccel

            self.xgyrozero += xgyro
            self.ygyrozero += ygyro
            self.zgyrozero += zgyro
            
        time2 = time.clock()
        i += 1
        ###T = (time2-time1)/i
        print 'time frame = %f' %T
    
        xaccavg /= (i)
        yaccavg /= (i)
        zaccavg /= (i)
        self.xgyrozero /= (i)
        self.ygyrozero /= (i)
        self.zgyrozero /= (i)
        
        self.xacc = xaccavg - ACCEL_ZERO_G
        self.yacc = yaccavg - ACCEL_ZERO_G
        self.zacc = zaccavg - ACCEL_ZERO_G
        
            
    def __call__(self):
        xaccel, yaccel, zaccel, xgyro, ygyro, zgyro = self.Read_Data()
        # conversion to deg/s from gyro data sheets (IDG500 and LISY300AL)
        xgyrodeg = (xgyro - self.xgyrozero)/1.024 * AREF / 2. 
        ygyrodeg = (ygyro - self.ygyrozero)/1.024 * AREF / 2. 
        zgyrodeg = (zgyro - self.zgyrozero)/1.024 * AREF / 3.3
        xacc = xaccel - ACCEL_ZERO_G
        yacc = yaccel - ACCEL_ZERO_G
        zacc = zaccel - ACCEL_ZERO_G
        return xacc, yacc, zacc, xgyrodeg, ygyrodeg, zgyrodeg
        
# rotating box driven by gyro sensors only
##mybox =box(pos = vector (0,0,-1),color = color.blue, axis=(1.,0.,0.))
##label (pos=(0,1.,0), text = 'Gyro Only')
##freezelabel = label(pos=(0,-1.,0), 
##    text = "POINT XBEE OR FDTI AT SCREEN\nDON'T MOVE BOARD YET")

# new window with box driven by complimentary filter (and gyro-only in yaw)
##scene2 = display(x = 500)
##mybox2 =box(pos = vector (0,0,-1),color = color.red, axis=(1.,0.,0.))
##label (pos=(0,1.,0), text = 'Comp Filter')
##freezelabel2 = label(pos=(0,-1.,0), 
##    text = "POINT XBEE OR FDTI AT SCREEN\nDON'T MOVE BOARD YET")


# file to debug code
###file = open("C:\\data.csv",'w')
###file.write("(-xgyrodeg),(-ygyrodeg),zgyrodeg,\
###x_comp_filter,y_comp_filter,xacc,yacc,(-zacc),anglez\n")


# open Xbee serial port input and initialize sensors
input_data = Process_Pushpak_Data(Com_Port = '/dev/cu.usbmodem1411',Baud = 115200)

xacc = input_data.xacc
yacc = input_data.yacc
zacc = input_data.zacc

one_gee = math.sqrt(xacc**2 + yacc**2 + zacc**2)

# turn off "don't move board" message
##freezelabel.visible = 0
##freezelabel2.visible = 0

# initialize comp filters

anglez = 0.

my_x_comp_filter = ComplementaryFilter(SamplePeriod = T,BandWidth = k,
                    Gyro_zero = 0., On_Axis_zero = yacc,Off_Axis_Zero=xacc,
                    Z_Axis_zero=zacc,One_Gee=one_gee)
 
my_y_comp_filter = ComplementaryFilter(SamplePeriod = T,BandWidth = k,
                    Gyro_zero = 0., On_Axis_zero = xacc,Off_Axis_Zero=zacc,
                    Z_Axis_zero=zacc,One_Gee=one_gee)

# get intial dataset to initialize the gyro-only cube
xacc, yacc, zacc, xgyrodeg, ygyrodeg, zgyrodeg = input_data()

x_comp_filter = my_x_comp_filter(Gyro_input=math.radians(xgyrodeg),
                    On_Axis_input=yacc, Off_Axis_input=xacc,Z_Axis_input=zacc)

y_comp_filter = my_y_comp_filter(Gyro_input=math.radians(ygyrodeg),
                    On_Axis_input=xacc, Off_Axis_input=yacc,Z_Axis_input=zacc)


# initialize gyro-only cube to comp filter cube position (see comments below)
##up_copy = mybox.up
##mybox.rotate(angle=y_comp_filter, axis = cross(mybox.axis,mybox.up))
##up_copy.rotate(angle=y_comp_filter, axis = cross(mybox.axis,mybox.up))
##mybox.rotate(angle=x_comp_filter, axis = mybox.axis)
##up_copy.rotate(angle=x_comp_filter, axis = mybox.axis)
##mybox2.rotate(angle=anglez, axis = up_copy)

compX=[]
compY=[]
cnt = 0

# spin cubes based on integrating raw gyro and on comp filter
try:
    while True:
        
        xacc, yacc, zacc, xgyrodeg, ygyrodeg, zgyrodeg = input_data()

        dtx = -math.radians(xgyrodeg * T)
        dty = math.radians(ygyrodeg * T)
        dtz = -math.radians(zgyrodeg * T)
        
        
        x_comp_filter = my_x_comp_filter(Gyro_input=math.radians(xgyrodeg),
                        On_Axis_input=yacc, Off_Axis_input=xacc,Z_Axis_input=zacc)

        y_comp_filter = my_y_comp_filter(Gyro_input=math.radians(ygyrodeg),
                        On_Axis_input=xacc, Off_Axis_input=yacc,Z_Axis_input=zacc)

        compX.append(float(x_comp_filter))
        drawnow(makeFig)
        compY.append(y_comp_filter)

        print(compX[cnt])

        # There is no comp filter in yaw
        # Put a deadzone of about 1.5 LSB to supress drift (may not work in-flight)
        if abs(math.radians(zgyrodeg)) > 0.03:
            anglez += dtz

        cnt = cnt + 1
except KeyboardInterrupt:
    pass

datacnt = 0

with open(savepath+file_name+'.csv', 'wb') as csvfile:
    print 'Saving Data...'
    while(datacnt<len(compX)):
        dataWriter = csv.writer(csvfile)
        dataWriter.writerow([compX[datacnt]])
        datacnt = datacnt + 1          
# for debugging and data collection 
###    file.write('%f,%f,%f,%f,%f,%f,%f,%f,%f\n' % 
###            (-math.radians(xgyrodeg),-math.radians(ygyrodeg),math.radians(zgyrodeg),
###            x_comp_filter,y_comp_filter,xacc,yacc,-zacc,anglez))

    #Drive right hand cube from comp filter output. Since this uses absolute 
    #angles, we turn off the cube, reset it to nominal, and rotate it by the
    #appropriate amount around each axis
    
##    mybox2.visible = 0
##    mybox2.axis = (1.,0.,0.) # cube's x-axis
##    mybox2.up = (0.,1.,0.) # cube's z-axis
##    up_copy = mybox2.up # vpython keeps the "up" axis pointing up, so we'll
                        # copy it so we can keep it oriented with the cube
    
    # cross(mybox2.axis,mybox2.up) is the cube's y-axis
##    mybox2.rotate(angle=y_comp_filter, axis = cross(mybox2.axis,mybox2.up))
##    up_copy.rotate(angle=y_comp_filter, axis = cross(mybox2.axis,mybox2.up))
##    mybox2.rotate(angle=x_comp_filter, axis = mybox2.axis)
##    up_copy.rotate(angle=x_comp_filter, axis = mybox2.axis)
##    mybox2.rotate(angle=anglez, axis = up_copy)

##    mybox2.visible = 1


    # using raw rates to drive cube
    # for any vector V, dV/dt = omega x V, so V = V + dV = V + (omega x V) dt
##    mybox.axis = norm(mybox.axis + cross((-dtx,dtz,-dty),mybox.axis))
    # then rotate about the x-axis
##    mybox.rotate(angle = dtx, axis = mybox.axis)

