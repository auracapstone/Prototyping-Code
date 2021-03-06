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

###################################################################

timecount=1

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

#####################################################################
#Main Functions





#####################################################################

# make the real time figure
def makeFig():
    plt.plot(compX,'r-')

#####################################################################

# peak detection

def peakdet(v, delta, x = None):
    """
    Converted from MATLAB script at http://billauer.co.il/peakdet.html
    
    Returns two arrays
    
    function [maxtab, mintab]=peakdet(v, delta, x)
    %PEAKDET Detect peaks in a vector
    %        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
    %        maxima and minima ("peaks") in the vector V.
    %        MAXTAB and MINTAB consists of two columns. Column 1
    %        contains indices in V, and column 2 the found values.
    %      
    %        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
    %        in MAXTAB and MINTAB are replaced with the corresponding
    %        X-values.
    %
    %        A point is considered a maximum peak if it has the maximal
    %        value, and was preceded (to the left) by a value lower by
    %        DELTA.
    
    % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    % This function is released to the public domain; Any use is allowed.
    
    """
    maxtab = []
    mintab = []
       
    if x is None:
        x = arange(len(v))
    
    v = asarray(v)
    
    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')
    
    if not isscalar(delta):
        sys.exit('Input argument delta must be a scalar')
    
    if delta <= 0:
        sys.exit('Input argument delta must be positive')
    
    mn, mx = Inf, -Inf
    mnpos, mxpos = NaN, NaN
    
    lookformax = True
    
    for i in arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]
        
        if lookformax:
            if this < mx-delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

    return array(maxtab), array(mintab)

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
        
# Define routines to processes Pushpak data. Alter this to suit your on-board 
# data formats. My Pushpak puts out 7 comma separated sets of ASCII numbers 
# (timeframe and 6 sensors). The sensor outputs are raw (uncalibrated)  
# 15x oversampled ADC values.
class Process_Pushpak_Data:
    previous_input = "0,0,0,0,0,0,0\n"

    def Close_Serial(self):
        print ('hello')
    
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
        return timeframe, xaccel, yaccel, zaccel, xgyro, ygyro, zgyro
        
    def __init__(self,Com_Port,Baud):
        #open serial port
        self.ser = serial.Serial(Com_Port,Baud) #to open COM4, use value 3.
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
        for i in range(300):
            timeframe, xaccel, yaccel, zaccel, xgyro, ygyro, zgyro = self.Read_Data()
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
        timeframe, xaccel, yaccel, zaccel, xgyro, ygyro, zgyro = self.Read_Data()
        # conversion to deg/s from gyro data sheets (IDG500 and LISY300AL)
        xgyrodeg = (xgyro - self.xgyrozero)/1.024 * AREF / 2. 
        ygyrodeg = (ygyro - self.ygyrozero)/1.024 * AREF / 2. 
        zgyrodeg = (zgyro - self.zgyrozero)/1.024 * AREF / 3.3
        xacc = xaccel - ACCEL_ZERO_G
        yacc = yaccel - ACCEL_ZERO_G
        zacc = zaccel - ACCEL_ZERO_G
        return timeframe, xacc, yacc, zacc, xgyrodeg, ygyrodeg, zgyrodeg
        

input_data = Process_Pushpak_Data(Com_Port = comPort,Baud = 115200)
xacc = input_data.xacc
yacc = input_data.yacc
zacc = input_data.zacc



one_gee = math.sqrt(xacc**2 + yacc**2 + zacc**2)

anglez = 0

my_x_comp_filter = ComplementaryFilter(SamplePeriod = T,BandWidth = k,
                    Gyro_zero = 0., On_Axis_zero = yacc,Off_Axis_Zero=xacc,
                    Z_Axis_zero=zacc,One_Gee=one_gee)
 
my_y_comp_filter = ComplementaryFilter(SamplePeriod = T,BandWidth = k,
                    Gyro_zero = 0., On_Axis_zero = xacc,Off_Axis_Zero=zacc,
                    Z_Axis_zero=zacc,One_Gee=one_gee)

timeframe, xacc, yacc, zacc, xgyrodeg, ygyrodeg, zgyrodeg = input_data()

x_comp_filter = my_x_comp_filter(Gyro_input=math.radians(xgyrodeg),
                    On_Axis_input=yacc, Off_Axis_input=xacc,Z_Axis_input=zacc)

y_comp_filter = my_y_comp_filter(Gyro_input=math.radians(ygyrodeg),
                    On_Axis_input=xacc, Off_Axis_input=yacc,Z_Axis_input=zacc)


compX=[]
compY=[]


try:
    cnt = 0
    while True:
		
		timeframe, xacc, yacc, zacc, xgyrodeg, ygyrodeg, zgyrodeg = input_data()

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

		if abs(math.radians(zgyrodeg)) > 0.03:
			anglez += dtz
		
		print timeframe
		
        if timeframe > 3000*timecount:
                series=compX
                ser.close()
                maxtab, mintab = peakdet(series,.001)
                print maxtab
                timecount=timecount+1

		cnt = cnt + 1
except KeyboardInterrupt:
    pass


with open(savepath+file_name+'.csv', 'wb') as csvfile:
    datacnt = 0
    print 'Saving Data...'
    while(datacnt<len(compX)):
        dataWriter = csv.writer(csvfile)
        dataWriter.writerow([compX[datacnt]])
        datacnt = datacnt + 1          
