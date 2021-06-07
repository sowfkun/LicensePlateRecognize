import cv2
import time
from absl.flags import FLAGS
from absl import app, flags, logging

flags.DEFINE_string('streamurl', 'http://192.168.1.160:8080/video', 'path to input video')

# main Function
def main(_argv):
    streamurl = FLAGS.streamurl
    frame_index = 1

    is_connected = False
    while True:
    
        if not is_connected:
            vid = cv2.VideoCapture(streamurl)
            
        return_value, frame = vid.read()
        if return_value:
            cv2.imwrite("./input/frame_" + str(frame_index) + ".jpg" , frame)
            frame_index += 1
            time.sleep(0.4) # 1s take three frame
            is_connected = True
        else:
            is_connected = False 

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass

