import cv2
import time
import random
import colorsys
import numpy as np
import tensorflow as tf
import core.utils as utils
import imutils
import re
import requests

from PIL import Image
from core.config import cfg
from skimage.filters import threshold_local
from skimage import measure

# Funtion draw box bounding plate
def draw_bbox(image, bboxes, show_label=True):
    num_classes = 1
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    out_boxes, out_scores, out_classes, num_boxes = bboxes
    list_plates = []
    for i in range(num_boxes[0]):
        if int(out_classes[0][i]) < 0 or int(out_classes[0][i]) > num_classes: continue
        coor = out_boxes[0][i]
        coor[0] = int(coor[0] * image_h)
        coor[2] = int(coor[2] * image_h)
        coor[1] = int(coor[1] * image_w)
        coor[3] = int(coor[3] * image_w)

        fontScale = 0.5
        score = out_scores[0][i]
        class_ind = int(out_classes[0][i])
        bbox_color = colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (coor[1], coor[0]), (coor[3], coor[2])
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)
        
        crop_plate = image[int(coor[0]):int(coor[2]), int(coor[1]):int(coor[3])]
        crop_plate_rgb = cv2.cvtColor(np.array(crop_plate), cv2.COLOR_BGR2RGB)
        list_plates.append(crop_plate_rgb)

    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)  
    return image, list_plates 

# Convert images to square image
def convertToSquare(image):
    """
    Resize non square image(height != width to square one (height == width)
    :param image: input images
    :return: numpy array
    """

    img_h = image.shape[0]
    img_w = image.shape[1]

    # if height > width
    if img_h > img_w:
        diff = img_h - img_w
        if diff % 2 == 0:
            x1 = np.zeros(shape=(img_h, diff//2))
            x2 = x1
        else:
            x1 = np.zeros(shape=(img_h, diff//2))
            x2 = np.zeros(shape=(img_h, (diff//2) + 1))

        squared_image = np.concatenate((x1, image, x2), axis=1)
    elif img_w > img_h:
        diff = img_w - img_h
        if diff % 2 == 0:
            x1 = np.zeros(shape=(diff//2, img_w))
            x2 = x1
        else:
            x1 = np.zeros(shape=(diff//2, img_w))
            x2 = x1

        squared_image = np.concatenate((x1, image, x2), axis=0)
    else:
        squared_image = image

    return squared_image

def filter_boxes(box_xywh, scores, score_threshold=0.4, input_shape = tf.constant([416,416])):
    scores_max = tf.math.reduce_max(scores, axis=-1)

    mask = scores_max >= score_threshold
    class_boxes = tf.boolean_mask(box_xywh, mask)
    pred_conf = tf.boolean_mask(scores, mask)
    class_boxes = tf.reshape(class_boxes, [tf.shape(scores)[0], -1, tf.shape(class_boxes)[-1]])
    pred_conf = tf.reshape(pred_conf, [tf.shape(scores)[0], -1, tf.shape(pred_conf)[-1]])

    box_xy, box_wh = tf.split(class_boxes, (2, 2), axis=-1)

    input_shape = tf.cast(input_shape, dtype=tf.float32)

    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]

    box_mins = (box_yx - (box_hw / 2.)) / input_shape
    box_maxes = (box_yx + (box_hw / 2.)) / input_shape
    boxes = tf.concat([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ], axis=-1)
    # return tf.concat([boxes, pred_conf], axis=-1)
    return (boxes, pred_conf)

# Function process detect plate from the frame
def frame_analyze(frame, tf, recogChar, iou, score, input_size):

    # Load TFLite weight
    interpreter = tf.lite.Interpreter(model_path='./TFLites/DetectPlate.tflite')
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Convert Color, resize frame,...
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame)
    frame_size = frame.shape[:2]
    image_data = cv2.resize(frame, (input_size, input_size))
    image_data = image_data / 255.
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    # using output detail from weight to detect
    interpreter.set_tensor(input_details[0]['index'], image_data)
    interpreter.invoke()
    pred = [interpreter.get_tensor(output_details[i]['index'])for i in range(len(output_details))]
    boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25, input_shape=tf.constant([input_size, input_size]))

    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(pred_conf, (tf.shape(pred_conf)[
                          0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=iou,
        score_threshold=score
    )

    # If object Detected, valid_detection is 1 else 0
    # If 0 skip, else draw box, then detect the number
    if valid_detections.numpy()[0] != 0:
        # draw box
        pred_bbox     = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
        draw_box      = utils.draw_bbox(frame, pred_bbox)
        image         = draw_box[0]
        list_plate    = draw_box[1]
        list_plate_no = plate_analyze(list_plate, recogChar)
        
        return (image, list_plate_no)
    else:
        return "no_plate"
        
# Function analyze plate to identify plate number
def plate_analyze(list_plate, recogChar):
    ALPHA_DICT = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'K', 9: 'L', 10: 'M', 11: 'N', 12: 'P',
              13: 'R', 14: 'S', 15: 'T', 16: 'U', 17: 'V', 18: 'X', 19: 'Y', 20: 'Z', 21: '0', 22: '1', 23: '2', 24: '3',
              25: '4', 26: '5', 27: '6', 28: '7', 29: '8', 30: '9', 31: "Background"}

    list_plate_no = []
    for plate in list_plate:
        #convert sang mau hsv
        plate_hsv = cv2.split(cv2.cvtColor(plate, cv2.COLOR_RGB2HSV))[2]
        # sử dụng adaptive threshold để làm nổi bật (màu đen)
        T = threshold_local(plate_hsv, 15, offset=10, method="gaussian")
        thresh = (plate_hsv > T).astype("uint8") * 255
        # convert black pixel of digits to white pixel
        thresh = cv2.bitwise_not(thresh)
        thresh = imutils.resize(thresh, width=400)
        thresh = cv2.medianBlur(thresh, 5)
        # Combine pixels have same value to blocks and label these blocks
        labels = measure.label(thresh, connectivity=2, background=0)
        cv2.imwrite("output/plate.jpg", thresh)

        # Filter unexpected label
        img_position_list = []
        for label in np.unique(labels):
            # if this is background label, ignore it
            if label == 0:
                continue
            
            # init mask to store the location of the character candidates
            mask = np.zeros(thresh.shape, dtype="uint8")
            mask[labels == label] = 255
 
            # find contours from mask
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) > 0:
                contour      = max(contours, key=cv2.contourArea)
                (x, y, w, h) = cv2.boundingRect(contour)
                
                # rule to determine characters
                aspectRatio  = w / float(h)
                solidity     = cv2.contourArea(contour) / float(w * h)
                heightRatio  = h / float(plate.shape[0])

                if 0.1 < aspectRatio < 1.0 and solidity > 0.1 and 0.35 < heightRatio < 2.0:
                    img         = np.array(mask[y-5:y + h +5, x -5:x + w + 5])
                    img         = utils.convertToSquare(img)
                    img_resize  = cv2.resize(img, (28, 28), cv2.INTER_AREA)
                    img_reshape = img_resize.reshape((-1, 28, 28, 1))
                    img_position_list.append((img_reshape, (x, y, w, h)))
                    
        # Recognize character in image
        char_position_list = []        
        for img_position in img_position_list:
                char_img   = np.array(img_position[0])
                result     = recogChar.predict_on_batch(char_img)
                result_idx = np.argmax(result, axis=1)

                for i in range(len(result_idx)):
                    if result_idx[i] == 31:    # if is background or noise, ignore it
                        continue
                    char = ALPHA_DICT[result_idx[i]]
                    char_position_list.append((char, img_position[1]))

        # Sort the character from left to right, top to bottom
        line1 = []
        line2 = []
        if len(char_position_list) > 0:
            flag = char_position_list[0]
            line1_is_upper = True
            # Compare position of tow character
            # If the difference not greater than 40 then they are same line (may be not work with image has high inclination )
            for char, position in char_position_list: 
                if abs(position[1] - flag[1][1]) < 40:
                    line1.append((char, position))
                else:
                    line2.append((char, position))

            # If length of line 2 = 0 mean just 1 line
            # If Position of line 1 < line 2 mean line 1 is upper 
            if len(line2) != 0:
                if line1[0][1][1] > line2[0][1][0]:
                    line1_is_upper = False

            # Sort character by second element (x coordinate)
            def take_second(ele):
                return ele[1]
            line1 = sorted(line1, key=take_second)
            line2 = sorted(line2, key=take_second) 
            
            # Combine 2 line to create plate_no
            plate_no = ""
            if line1_is_upper:
                for char, position in line1:
                    plate_no += char
                for char, position in line2:
                    plate_no += char
            else:
                for char, position in line2:
                    plate_no += char
                for char, position in line1:
                    plate_no += char
            list_plate_no.append(re.sub(r'[\W_]+', '', plate_no))
    
    return list_plate_no
        
# Function Check Valid plate number
def plate_no_validation(list_plate_no):
    valid_plate_no = []
    for plate_no in list_plate_no:
        if 6 < len(plate_no) < 10:
            # tính từ 0
            # vị trí thứ 2 là kí tự => nếu 8 đổi thành B
            if plate_no[2] == "8":
                plate_no = plate_no[:2] + "B" + plate_no[2+1:]            
            
            # vị trí thứ 2 là kí tự => nếu 4 đổi thành A
            if plate_no[2] == "4":
                plate_no = plate_no[:2] + "A" + plate_no[2+1:]

            # vị trí thứ 2 là kí tự => nếu 0 đổi thành D
            if plate_no[2] == "0":
               plate_no = plate_no[:2] + "D" + plate_no[2+1:]
            
            # Biển LD => Nếu vị trí "L0" đổi thành LD
            if plate_no[2] == "L" and plate_no[3] == "0":
               plate_no = plate_no[:3] + "D" + plate_no[3+1:]
            
            valid_plate_no.append(plate_no)
        else: 
            continue
    return valid_plate_no 

# Post result to server
def post_to_server(image, list_plate_no):
    img_name = ""
    for plate in list_plate_no: 
        img_name = img_name + plate + "_"

    result_path = "./result/" + img_name[:-1] + ".jpg"
    cv2.imwrite(result_path, image)
    url  = 'http://vsscam.tk/api/ProcessDataFromEdge' 
    
    file = {"image" : open(result_path, "rb")}
    data = { "listplate": list_plate_no}
    print(list_plate_no)
    try:
        requests.post(url, files=file, data=data)
    except Exception as ex:
        print(ex)
    

def get_digits_data(path):
    data = np.load(path, allow_pickle=True)
    total_nb_data = len(data)
    np.random.shuffle(data)
    data_train = []

    for i in range(total_nb_data):
        data_train.append(data[i])

    print("-------------DONE------------")
    print('The number of train digits data: ', len(data_train))

    return data_train


def get_alphas_data(path):
    data = np.load(path, allow_pickle=True)
    total_nb_data = len(data)

    np.random.shuffle(data)
    data_train = []

    for i in range(total_nb_data):
        data_train.append(data[i])

    print("-------------DONE------------")
    print('The number of train alphas data: ', len(data_train))

    return data_train

