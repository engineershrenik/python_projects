#!/usr/bin/python

#https://github.com/engineershrenik/

# This code has following depedencies
# Python Mozilla webthings
# OpenCV2 module
# Gateway running Mozilla webthings
# Camera or Video input
#
# It has simple Std Deviation used for motion detection and then motion sensor
# Device type is used to make it compatible with Mozilla webthings
#

from __future__ import division, print_function
import numpy as np
import cv2
from webthing import (Action, Event, SingleThing, Property, Thing, Value,
                      WebThingServer)
import logging
import random
import time
import tornado.ioloop
import uuid
import threading
import asyncio

motion_status = 0


#webthings code
class OpenCvMotionSensor(Thing):

    def __init__(self):
        Thing.__init__(
                self,
                'urn:dev:ops:my-maaxboard-opencv-motion-sensor',
                'OpenCV motion sensor',
                ['MotionSensor'],
                'On device image processing for motion detection on MaaXBoard'
                )

        self.motion = Value(0)
        self.add_property(
                Property(self,
                    'motion',
                    self.motion,
                    metadata={
                        '@type': 'MotionProperty ',
                        'title': 'MaaxBoard Opencv Motion',
                        'type': 'boolean',
                        'description': 'Whether the motion detected',
                        'readOnly': True,
                        }))

        self.timer = tornado.ioloop.PeriodicCallback(
            self.update_motion,
            1000
        )
        self.timer.start()

    def update_motion(self):
        logging.debug('motion_status %s', motion_status)
        self.motion.notify_of_external_update(motion_status)

    def cancel_update_motion_task(self):
        self.timer.stop()


def run_server(name):
    asyncio.set_event_loop(asyncio.new_event_loop())
    # Create a thing that represents a opencv motion sensor
    sensor = OpenCvMotionSensor()

    server = WebThingServer(SingleThing(sensor), port=8888)
    try:
        logging.info('starting the server')
        server.start()
    except KeyboardInterrupt:
        logging.debug('canceling the sensor update looping task')
        sensor.cancel_update_level_task()
        logging.info('stopping the server')
        server.stop()
        logging.info('done')

x = threading.Thread(target=run_server, args=(1,))
x.start()

STD_DEV_THRESHOLD = 5
font = cv2.FONT_HERSHEY_SIMPLEX

def distMap(frame1, frame2):
    """outputs pythagorean distance between two frames"""
    frame1_32 = np.float32(frame1)
    frame2_32 = np.float32(frame2)
    diff32 = frame1_32 - frame2_32
    norm32 = np.sqrt(diff32[:,:,0]**2 + diff32[:,:,1]**2 + diff32[:,:,2]**2)/np.sqrt(255**2 + 255**2 + 255**2)
    dist = np.uint8(norm32*255)
    return dist

cap = cv2.VideoCapture('./motion_test.mp4')

_, frame1 = cap.read()
_, frame2 = cap.read()

while(1):
    _, frame3 = cap.read()
    rows, cols, _ = np.shape(frame3)
    #cv2.imshow('dist', frame3)
    dist = distMap(frame1, frame3)

    frame1 = frame2
    frame2 = frame3

    # apply Gaussian smoothing
    mod = cv2.GaussianBlur(dist, (9,9), 0)

    # apply thresholding
    _, thresh = cv2.threshold(mod, 100, 255, 0)

    # calculate st dev test
    _, stDev = cv2.meanStdDev(mod)

    
    if stDev > STD_DEV_THRESHOLD:
        cv2.putText(mod,"Movement detected", (70, 70), font, 1, (255, 0, 255), 1, cv2.LINE_AA)
        motion_status = 1
    else:
        cv2.putText(mod, "No movement!!", (70, 70), font, 1, (255, 0, 255), 1, cv2.LINE_AA)
        motion_status = 0
    #    cv2.imshow('dist', mod)

    cv2.imshow('MaaXBoard motion detect', mod)
    #cv2.imshow('frame', frame2)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
