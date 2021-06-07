# Import libs
import os
import tensorflow as tf
import numpy as np
import cv2
import core.utils as utils
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

# main Function
def main(_argv):
    # Define some variable
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    input_size = 416
    iou = 0.45
    score = 0.25
    recogChar = CNN_Model(trainable=False).model
    recogChar.load_weights('./Weights/weight.h5')

    while True:
        if len(os.listdir("./input")) > 11:
            file = sorted(os.scandir('./input/'), key=lambda t: t.stat().st_mtime)[0]
            input_path = "./input/" + file.name
                
            img = cv2.imread(input_path)
                        
            analyze_result = utils.frame_analyze(img, tf, recogChar, iou, score, input_size)
                    
            if analyze_result != "no_plate":
                valid_plate_no = utils.plate_no_validation(analyze_result[1])        
                utils.post_to_server(analyze_result[0], valid_plate_no)
            else: 
                print("no plate")

            os.remove(input_path)  
            
            	           
if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass

