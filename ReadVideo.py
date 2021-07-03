import cv2
import time
from absl.flags import FLAGS
from absl import app, flags, logging

# main Function
def main(_argv):
    video_path = "./demo/video.mp4"
    vid = cv2.VideoCapture(video_path)
    
    frame_index = 1
    count = 0
    while True:
        return_value, frame = vid.read()
        if return_value:
            
            if count % 30 == 0:
                #frame = cv2.rotate(frame, cv2.ROTATE_180)
                cv2.imwrite("./input/frame_" + str(frame_index) + ".jpg" , frame)
                frame_index += 1
            count +=1 
        else:
            break
        

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass

