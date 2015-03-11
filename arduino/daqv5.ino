#include <Wire.h>
#include <OneWire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_LSM303_U.h>
#include <Adafruit_L3GD20_U.h>
#include <Adafruit_9DOF.h>

/* Assign a unique ID to the sensors */
Adafruit_LSM303_Accel_Unified accel = Adafruit_LSM303_Accel_Unified(30301);
Adafruit_L3GD20_Unified       gyro  = Adafruit_L3GD20_Unified(20);

int DS18S20_Pin = 2; //DS18S20 Signal pin on digital 2 - for thermistor
//Temperature chip i/o
OneWire ds(DS18S20_Pin);  // on digital pin 2

void setup(void)
{
  Serial.begin(115200);
  
  /* Initialise the sensors */
  if(!accel.begin())
  {
    /* There was a problem detecting the ADXL345 ... check your connections */
    Serial.println(F("Ooops, no LSM303 detected ... Check your wiring!"));
    while(1);
  }

  if(!gyro.begin())
  {
    /* There was a problem detecting the L3GD20 ... check your connections */
    Serial.print("Ooops, no L3GD20 detected ... Check your wiring or I2C ADDR!");
    while(1);
  }
  
  
}

void loop(void)
{
  /* Get a new sensor event */
  sensors_event_t event;
  
  /* Get temperature */
  float thermistorTemp = getTemp();
  
  /* Serial Print: time, accelX, accelY, accelXZ, gyrX, gyrY, gyrZ, temperature */
  Serial.print(millis()); Serial.print(",");
  
  accel.getEvent(&event);
  Serial.print(event.acceleration.x); Serial.print(",");
  Serial.print(event.acceleration.y); Serial.print(",");
  Serial.print(event.acceleration.z); Serial.print(",");
  
  gyro.getEvent(&event);
  Serial.print(event.gyro.x); Serial.print(",");
  Serial.print(event.gyro.y); Serial.print(",");
  Serial.println(event.gyro.z);
  
  Serial.print(thermistorTemp); Serial.print(",");

  delay(50);
}

/* thermistor temperature function */

float getTemp(){
  //returns the temperature from one DS18S20 in DEG Celsius

  byte data[12];
  byte addr[8];

  if ( !ds.search(addr)) {
      //no more sensors on chain, reset search
      ds.reset_search();
      return -1000;
  }

  if ( OneWire::crc8( addr, 7) != addr[7]) {
      Serial.println("CRC is not valid!");
      return -1000;
  }

  if ( addr[0] != 0x10 && addr[0] != 0x28) {
      Serial.print("Device is not recognized");
      return -1000;
  }

  ds.reset();
  ds.select(addr);
  ds.write(0x44,1); // start conversion, with parasite power on at the end

  byte present = ds.reset();
  ds.select(addr);    
  ds.write(0xBE); // Read Scratchpad

  
  for (int i = 0; i < 9; i++) { // we need 9 bytes
    data[i] = ds.read();
  }
  
  ds.reset_search();
  
  byte MSB = data[1];
  byte LSB = data[0];

  float tempRead = ((MSB << 8) | LSB); //using two's compliment
  float TemperatureSum = tempRead / 16;

  
  return TemperatureSum;
  
}
