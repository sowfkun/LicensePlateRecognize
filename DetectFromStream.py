# Import libs
import os
import tensorflow as tf
import numpy as np
import cv2
import core.utils as utils
import shutil
import time
from absl.flags import FLAGS
from absl import app, flags, logging
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.compat.v1 import ConfigProto
from threading import Thread
from core.model import CNN_Model
from pathlib import Path

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


# Define thread
def detect_threading(os, img_index, tf, recogChar, iou, score, input_size):
    
    file = sorted(os.scandir('./input/'), key=lambda t: t.stat().st_mtime)[img_index]
    input_path = "./input/" + file.name
    processing_path = "./processing/" + file.name
        
    img = cv2.imread(input_path)
    cv2.imwrite(processing_path, img)
                
    analyze_result = utils.frame_analyze(img, tf, recogChar, iou, score, input_size)
            
    if analyze_result != "no_plate":
        valid_plate_no = utils.plate_no_validation(analyze_result[1])        
        utils.post_to_server(analyze_result[0], valid_plate_no)
    else: 
        print("no plate")
    
    os.remove(input_path)

# main Function
def main(_argv):
    # Define some variable
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = 416
    iou = 0.45
    score = 0.25
    recogChar = CNN_Model(trainable=False).model
    recogChar.load_weights('./Weights/weight.h5')
       
    while True:
        if len(os.listdir("./input")) > 5:
            thread1 = Thread(target=detect_threading, args=(os, 0, tf, recogChar, iou, score, input_size))
            thread2 = Thread(target=detect_threading, args=(os, 1, tf, recogChar, iou, score, input_size))
            thread3 = Thread(target=detect_threading, args=(os, 2, tf, recogChar, iou, score, input_size))
            thread4 = Thread(target=detect_threading, args=(os, 3, tf, recogChar, iou, score, input_size))
            
            thread1.start()
            thread2.start()
            thread3.start()
            thread4.start()
            
            thread1.join()
            thread2.join()
            thread3.join()
            thread4.join()
            	           
if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass

