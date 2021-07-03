import cv2
import numpy as np
from absl.flags import FLAGS
from absl import app, flags, logging

flags.DEFINE_string('streamurl', 'http://192.168.1.7:8080/video', 'path to input video')


def main(_argv):
  # Create a VideoCapture object and read from input file
  # If the input is the camera, pass 0 instead of the video file name
  streamurl = FLAGS.streamurl
  cap = cv2.VideoCapture(streamurl)
  # Check if camera opened successfully
  if (cap.isOpened()== False):
    print("Error opening video stream or file")
  # Read until video is completed
  frame_index = 1
  count = 0
  while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
      # Display the resulting frame
      cv2.imshow('Frame',frame)
      if count % 30 == 0:
        #frame = cv2.rotate(frame, cv2.ROTATE_180)
        cv2.imwrite("./input/frame_" + str(frame_index) + ".jpg" , frame)
        frame_index += 1      
      count += 1
      # Press Q on keyboard to  exit
      if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    # Break the loop
    else:
      break
  # When everything done, release the video capture object
  cap.release()
  # Closes all the frames
  cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
