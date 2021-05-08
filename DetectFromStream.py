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

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Define flags
flags.DEFINE_string('streamurl', 'http://192.168.1.160:8080/video', 'path to input video')

# main Function
def main(_argv):
    # Define some variable
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = 416
    streamurl = FLAGS.streamurl
    iou = 0.45
    score = 0.25
    recogChar = CNN_Model(trainable=False).model
    recogChar.load_weights('./Weights/weight.h5')

    # Load TFLite weight
    interpreter = tf.lite.Interpreter(model_path='./TFLites/DetectPlate.tflite')
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Capture video from the url
    vid = cv2.VideoCapture(streamurl)
    while True:
        # Read frame from the stream
        return_value, frame = vid.read()
        if return_value:
            analyze_result = utils.frame_analyze(frame, interpreter, input_details, output_details, recogChar, iou, score, input_size)
            
            if analyze_result != "no_plate":
                valid_plate_no = utils.plate_no_validation(analyze_result[1])        
                utils.post_to_server(analyze_result[0], valid_plate_no)
        else:
            break
            
if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
