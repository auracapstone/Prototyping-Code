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
#from __future__ import division
import numpy as np

###################################################################

#File Saving and User Definitions 
file_name = raw_input('save file name without spaces!! : ')
developer = raw_input('are you rushi or hannelle? : ')

if (developer=="rushi" or developer=="Rushi" or developer=="r"):
    comPort = '/dev/cu.usbmodem1411'
    savepath = '/Users/rushipatel/Documents/SeniorDesignApps/PythonSscripts/'
    
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
time2 = []
accX = []
accY = []
accZ = []
gyrX = []
gyrY = []
gyrZ = []
ind = []
plot =[]
filteredX = []
filteredY = []
peakX = []
peakY = []
fftX = []
freqX = []
cnt2 = 0

#######################################################################


def ourFFT(X,Y,time):
    fftX = np.fft.fft(X)
    fftY = np.fft.fft(Y)
    freqX = np.fft.fftfreq(len(time))
    print len(fftX)
    print len(freqX)
    return freqX, fftX, fftY

def detect_peaks(x, mph=None, mpd=20, threshold=0, edge='rising',
                 kpsh=False, valley=False, show=False, ax=None):

    """Detect peaks in data based on their amplitude and other features.

    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height.
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).

    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.

    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`
    
    The function can handle NaN's 

    See this IPython Notebook [1]_.

    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb

    Examples
    --------
    >>> from detect_peaks import detect_peaks
    >>> x = np.random.randn(100)
    >>> x[60:81] = np.nan
    >>> # detect all peaks and plot data
    >>> ind = detect_peaks(x, show=True)
    >>> print(ind)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # set minimum peak height = 0 and minimum peak distance = 20
    >>> detect_peaks(x, mph=0, mpd=20, show=True)

    >>> x = [0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0]
    >>> # set minimum peak distance = 2
    >>> detect_peaks(x, mpd=2, show=True)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # detection of valleys instead of peaks
    >>> detect_peaks(x, mph=0, mpd=20, valley=True, show=True)

    >>> x = [0, 1, 1, 0, 1, 1, 0]
    >>> # detect both edges
    >>> detect_peaks(x, edge='both', show=True)

    >>> x = [-2, 1, -2, 2, 1, 1, 3, 0]
    >>> # set threshold = 2
    >>> detect_peaks(x, threshold = 2, show=True)
    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
        _plot(x, mph, mpd, threshold, edge, valley, ax, ind)

    return ind


def _plot(x, mph, mpd, threshold, edge, valley, ax, ind):
    """Plot results of the detect_peaks function, see its help."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
    else:
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 4))

        ax.plot(x, 'b', lw=1)
        if ind.size:
            label = 'valley' if valley else 'peak'
            label = label + 's' if ind.size > 1 else label
            ax.plot(ind, x[ind], '+', mfc=None, mec='r', mew=2, ms=8,
                    label='%d %s' % (ind.size, label))
            ax.legend(loc='best', framealpha=.5, numpoints=1)
        ax.set_xlim(-.02*x.size, x.size*1.02-1)
        ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
        yrange = ymax - ymin if ymax > ymin else 1
        ax.set_ylim(ymin - 0.1*yrange, ymax + 0.1*yrange)
        ax.set_xlabel('Data #', fontsize=14)
        ax.set_ylabel('Amplitude', fontsize=14)
        mode = 'Valley detection' if valley else 'Peak detection'
        ax.set_title("%s (mph=%s, mpd=%d, threshold=%s, edge='%s')"
                     % (mode, str(mph), mpd, str(threshold), edge))
        # plt.grid()
        plt.show()
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
    plt.figure(1)
    plt.subplot(211)
    plt.plot(filteredX,'r-')
    for j in range(len(peakX)):
        plt.plot(peakX[j], filteredX[peakX[j]],'bo')

    plt.subplot(212)
    plt.plot(filteredY,'r-')
    for j in range(len(peakY)):
        plt.plot(peakY[j], filteredY[peakY[j]],'bo')

    plt.show()
#####################################################################

def makeaFig():
    plt.figure(2)
    plt.plot(freqX, fftX.real)

# make the real time figure
def resetVariables():
    accX = []
    accY = []
    accZ = []
    gyrX = []
    gyrY = []
    gyrZ = []

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

    for i in range(len(accX)):
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
                if (cnt2 > 3):
                    print time[cnt2] - time[cnt2-1]
                cnt2 = cnt2 + 1
                if(timecount>1):
                    time2.append(t)


            

                ##filtering 
                if(t>3000*timecount):
                    if (timecount == 1): 
                        my_x_comp_filter, my_y_comp_filter, accXavg, accYavg, accZavg, gyrXavg, gyrYavg, gyrZavg = initializeCompFilter(time,accX,accY,accZ,gyrX,gyrY,gyrZ) 
                        print accX
                        accX = []
                        accY = []
                        accZ = []
                        gyrX = []
                        gyrY = []
                        gyrZ = []
                    if (timecount > 1):
                        filteredX, filteredY = compFilter(my_x_comp_filter, my_y_comp_filter,time,accX,accY,accZ,gyrX,gyrY,gyrZ,gyrXavg, gyrYavg, gyrZavg)
                        accX = []
                        accY = []
                        accZ = []
                        gyrX = []
                        gyrY = []
                        gyrZ = []
                        freqX, fftX, fftY = ourFFT(filteredX,filteredY,time2)
						maxsignalX=fftx(freqX[0:3])
						maxfreqX=np.amax(maxsignalX)
						RespRate = maxfreqX*3000*timecount
                        drawnow(makeFig)
                        drawnow(makeaFig)
                        #peakX = detect_peaks(filteredX)
                        #peakY = detect_peaks(filteredY)
                        #if (len(peakX)>=1 & len(peakY)>=1):
                        #    drawnow(makeFig)
                    timecount=timecount+1
            cnt = cnt + 1
except KeyboardInterrupt:
    pass
