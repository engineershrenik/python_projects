# import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import serial

###### HW code ####
import RPi.GPIO as GPIO
from RPLCD import CharLCD
from pad4pi import rpi_gpio
import time
KEYPAD = [
    [1, 2, 3, "<"],
    [7, 8, 9, ">"],
    [4, 5, 6, " "],
    ["S", 0, "R", "L"]
]

ser = serial.Serial('/dev/ttyUSB0')
stored_card = '0007919086'
lcd = CharLCD(cols=16, rows=2, pin_rs=2, pin_e=3, pins_data=[4, 17, 27, 8], numbering_mode=GPIO.BCM)

ROW_PINS = [24, 9, 11, 10] # BCM numbering
COL_PINS = [25, 12, 7, 5] # BCM numbering

input_pw = [0, 0, 0, 0]
preset_pw = [2, 4, 6, 8]
factory = rpi_gpio.KeypadFactory()
keypad = factory.create_keypad(keypad=KEYPAD, row_pins=ROW_PINS, col_pins=COL_PINS)
pw_count = 0
pw_config = 0
def printKey(key):
    print(key)
    #lcd.clear()
    global pw_config
    global input_pw
    global preset_pw
    global pw_count

    if pw_config == 1:
        input_pw[pw_count] = key
        lcd.cursor_pos = (1, pw_count + 1)
        lcd.write_string('*');
        #pw_config = pw_config + 1
        if pw_count == 3:
            if input_pw ==  preset_pw:
                lcd.clear()
                lcd.write_string(u'login Sucess :)');
                print(u'login Sucess :)');
            else:
                lcd.clear()
                lcd.write_string(u'login Failed :(');
                print(u'login Failed :(');
                time.sleep(1)
            pw_config = 0
        pw_count = pw_count + 1
        print 'pw count:' + str(pw_count)
    if key == 'L':
        lcd.clear()
        lcd.write_string(u'Show RFID to login');
        card_no = ser.readline()
        lcd.clear()
        if card_no[1:11] == stored_card:
            lcd.write_string(u'login Sucess :)');
            print(u'login Success :(');
        else:
            lcd.write_string(u'login Failed :(');
            print(u'login Failed :(');
            time.sleep(1)
            lcd.clear()
            lcd.write_string(u'Try password')
            print(u'Try pw :(');
            time.sleep(1)
            #lcd.clear()
            #lcd.cursor_pos(2, 0)
            pw_config = 1


keypad.registerKeyPressHandler(printKey)
lcd.clear()
time.sleep(1)
lcd.write_string(u'Quality Control!')
time.sleep(1)
#number as per BCM GPIO and not board GPIO
MOTOR1_CW = 14
MOTOR1_CCW = 15
MOTOR2_CW = 18
MOTOR2_CCW = 23
IR_SENSOR1 = 6	#pin 29
IR_SENSOR2 = 13  #pin 31

GPIO.setwarnings(False)
GPIO.setmode (GPIO.BCM)

#Set GPIOs for MOTOR
GPIO.setup(MOTOR1_CW,GPIO.OUT)
GPIO.setup(MOTOR1_CCW,GPIO.OUT)
GPIO.setup(MOTOR2_CW,GPIO.OUT)
GPIO.setup(MOTOR2_CCW,GPIO.OUT)
#set GPIOs for IR sensors
GPIO.setup(IR_SENSOR1,GPIO.IN)
GPIO.setup(IR_SENSOR2,GPIO.IN)

GPIO.add_event_detect(IR_SENSOR1, GPIO.RISING)
GPIO.add_event_detect(IR_SENSOR2, GPIO.RISING)
ir1_val = 0
ir2_val = 0
def ir1_callback(self):
    global ir1_val
    ir1_val =  1
    print ('IR1 Detected!')
    
def ir2_callback(self):
    global ir2_val
    ir2_val =  1
    print ('IR2 Detected!')

GPIO.add_event_callback(IR_SENSOR1, ir1_callback)
GPIO.add_event_callback(IR_SENSOR2, ir2_callback)


###################


def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=False,
	help="path to the input image")
ap.add_argument("-w", "--width", type=float, required=True,
	help="width of the left-most object in the image (in inches)")
args = vars(ap.parse_args())


image_count = 0
found_bad_object = 0
# load the image, convert it to grayscale, and blur it slightly
#image = cv2.imread(args["image"])
def scan_object():
    cam = cv2.VideoCapture(0)
    ret, image = cam.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # perform edge detection, then perform a dilation + erosion to
    # close gaps in between object edges
    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    #cv2.imshow('edged', edged)
    # find contours in the edge map
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	    cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

        # sort the contours from left-to-right and initialize the
        # 'pixels per metric' calibration variable
    (cnts, _) = contours.sort_contours(cnts)
    pixelsPerMetric = None
    image_count = 0
    found_bad_object = 0

    # loop over the contours individually
    for c in cnts:
            # if the contour is not sufficiently large, ignore it
            if cv2.contourArea(c) < 100:
                    continue

            # compute the rotated bounding box of the contour
            orig = image.copy()
            box = cv2.minAreaRect(c)
            box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
            box = np.array(box, dtype="int")

            # order the points in the contour such that they appear
            # in top-left, top-right, bottom-right, and bottom-left
            # order, then draw the outline of the rotated bounding
            # box
            box = perspective.order_points(box)
            cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

            # loop over the original points and draw them
            for (x, y) in box:
                    cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

            # unpack the ordered bounding box, then compute the midpoint
            # between the top-left and top-right coordinates, followed by
            # the midpoint between bottom-left and bottom-right coordinates
            (tl, tr, br, bl) = box
            (tltrX, tltrY) = midpoint(tl, tr)
            (blbrX, blbrY) = midpoint(bl, br)

            # compute the midpoint between the top-left and top-right points,
            # followed by the midpoint between the top-righ and bottom-right
            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)

            # draw the midpoints on the image
            cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
            cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
            cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
            cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
            # draw lines between the midpoints
            cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                    (255, 0, 255), 2)
            cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                    (255, 0, 255), 2)

            # compute the Euclidean distance between the midpoints
            dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

            # if the pixels per metric has not been initialized, then
            # compute it as the ratio of pixels to supplied metric
            # (in this case, inches)
            if pixelsPerMetric is None:
                    pixelsPerMetric = dB / args["width"]

            # compute the size of the object
            dimA = dA / pixelsPerMetric
            dimB = dB / pixelsPerMetric
            #print ("below are dimentions")
            print ("dimA [" + str(dimA) + "] dimB[" + str(dimB) +"]")
            if ((dimA < .6 and dimA > 0.4) or (dimB < 1 and dimB > .6) and image_count > 0):
                    found_bad_object = found_bad_object + 1
                    print ("Bad Object")
            # draw the object sizes on the image
            cv2.putText(orig, "{:.1f}in".format(dimA),
                    (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2)
            cv2.putText(orig, "{:.1f}in".format(dimB),
                    (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2)
            image_count = image_count + 1

            # show the output image
            #cv2.imshow("Image", orig)
            #cv2.waitKey(0)

while 1:
    if ir1_val:
        time.sleep(1)
        GPIO.output(MOTOR1_CW, GPIO.LOW)
        ir1_val = 0
        scan_object()
    else:
        time.sleep(1)
        GPIO.output(MOTOR1_CW, GPIO.HIGH)
    if found_bad_object:
        GPIO.output(MOTOR1_CW, GPIO.HIGH)
        while ir2_val:
            time.sleep(.1)
        ir2_val = 0
        GPIO.output(MOTOR1_CW, GPIO.LOW)
        GPIO.output(MOTOR2_CW, GPIO.HIGH)
        GPIO.output(MOTOR2_CCW, GPIO.LOW)
        time.sleep(1)
        GPIO.output(MOTOR2_CCW, GPIO.HIGH)
        GPIO.output(MOTOR2_CW, GPIO.LOW)
        GPIO.output(MOTOR1_CW, GPIO.HIGH)
    else:
        GPIO.output(MOTOR2_CW, GPIO.LOW)
        GPIO.output(MOTOR1_CW, GPIO.HIGH)

   
    time.sleep(.5)

