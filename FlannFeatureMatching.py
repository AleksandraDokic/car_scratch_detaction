#===============================================================================
# author: Aleksandra Dokic
# data: 25.06.2018
# company: Telemotive AG
#===============================================================================
import cv2
import numpy as np
import os
from edge_detector import EdgeDetector

##################################
# gets thresholded image within specified colour range
##################################
def FindTheColour(img, lower, upper):

    # transform image to hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # create NumPy arrays from the boundaries
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")

    # find the colors within the specified boundaries and apply the mask
    colour_img = cv2.inRange(hsv, lower, upper)

    return colour_img

def ExtractTheRoi(img_colour, img_thresh):

    # define dialation kernel
    kernel = np.ones((5, 5), np.uint8)

    dilated_img = cv2.dilate(img_thresh, kernel, iterations=1)

    img_copy = img_colour.copy()

    # colour image with just the roi showed and everything else dark
    colour_masked = cv2.bitwise_and(img_copy, img_copy, mask=dilated_img)

    ret, thresh = cv2.threshold(dilated_img, 40, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    roi_big = img_colour.copy()
    height, width = roi_big.shape[:2]

    if len(contours) != 0:
        # draw in blue the contours that were founded
        cv2.drawContours(img_copy, contours, -1, 255, 3)

        # find the biggest area
        c = max(contours, key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(c)
        roi_big = roi_big[y:height, x:x + w]
        # draw the book contour (in green)
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

    #roi_big = cv2.Canny(roi_big, 100, 200)
    # show the images
    #cv2.imshow("Result", roi_big)

    return roi_big, x, y, w, h

def deleteOutTheTires(img_orange, img_black_tresholded):

    im2, contours, hierarchy = cv2.findContours(img_black_tresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea)

    roi_big = img_orange.copy()
    height, width = roi_big.shape[:2]

    if len(contours) != 0:
        # draw in blue the contours that were founded
        cv2.drawContours(img_orange, contours, -1, 255, 3)
        c1 = contours[-1]
        c2 = contours[-2]

        x, y, w, h = cv2.boundingRect(c1)
        x2, y2, w2, h2 = cv2.boundingRect(c2)

        # draw the book contour (in green)
        cv2.rectangle(img_orange, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(img_orange, (x2, y2), (x2 + w2, y2 + h2), (0, 0, 255), 2)

        if y > y2:
            roi_big = roi_big[0: y + h/2,0:width]
        else:
            roi_big = roi_big[0: y2 + h2/2, 0:width]

        return roi_big




def cutOutRoi(img):

    # define hsv range values for blue
    lower = [5, 100, 150]
    upper = [20, 255, 255]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # create NumPy arrays from the boundaries
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")

    # find the colors within the specified boundaries and apply
    # the mask
    imgExtractBlueThreshold = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((5, 5), np.uint8)
    imgExtractBlueThreshold = cv2.dilate(imgExtractBlueThreshold, kernel, iterations=1)

    cv2.namedWindow("Extracted orange", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Extracted orange", 400, 400)
    cv2.imshow("Extracted orange", imgExtractBlueThreshold)



    # find the colors within the specified boundaries and apply
    # the mask
    #mask = cv2.inRange(image, lower, upper)
    img_copy = img.copy()
    output = cv2.bitwise_and(img_copy, img_copy, mask=imgExtractBlueThreshold)

    cv2.imshow("Mhm", output)

    ret, thresh = cv2.threshold(imgExtractBlueThreshold, 40, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    roi_big = img.copy()
    height, width = roi_big.shape[:2]

    if len(contours) != 0:
        # draw in blue the contours that were founded
        cv2.drawContours(img_copy, contours, -1, 255, 3)

        # find the biggest area
        c = max(contours, key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(c)
        roi_big = roi_big[y:height, x:x + w]
        # draw the book contour (in green)
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
    else:
        print("No cntours")

    roi_big = cv2.Canny(roi_big, 100, 200)
    # show the images
    cv2.imshow("Result",roi_big)

    #self.imgExtractBlue = self.imgExtractBlueThreshold  # dummy-assignement to avoid crash on imshow!
    #self.imgExtractBlueThresholdOpened = self.fillSmallGaps(self.imgExtractBlueThreshold)

def extractTheCarArea(img):

    # define orange ranges
    lower = [5, 100, 150]
    upper = [20, 255, 255]

    img_orange_threshold = FindTheColour(img, lower, upper)
    img_orange_masked, x_o, y_o, w_o, h_o = ExtractTheRoi(img, img_orange_threshold)

    # define black ranges
    lowerB = [0, 0, 0]
    upperB = [180, 255, 80]
    img_black_threshold = FindTheColour(img_orange_masked, lowerB, upperB)

    img_final = deleteOutTheTires(img_orange_masked, img_black_threshold)

    #cv2.imshow("Thresholded", img_orange_threshold)
    cv2.imshow("Masked", img_orange_masked)
    #cv2.imshow("Black detection", img_black_threshold)
    cv2.imshow("Without tires", img_final)
    return img_final

'''def match_template(img_src, img_dst, template, threshold, color):

        # Store width and heigth of template in w and h
        w, h = template.shape[:2]

        # Perform match operations.
        res = cv2.matchTemplate(img_src, template, cv2.TM_CCOEFF_NORMED)
        # for TM_CCOEFF best match is max match
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # get match location in original image
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        # if match above threshold draw it
        if max_val > threshold:
            cv2.rectangle(img_dst, top_left, bottom_right, color, 2)
            return True
        return False'''

def define_random_point(img):
    pass

def process_video(filename, ed):

    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    cap = cv2.VideoCapture(filename)

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    # Read until video is completed
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:

            # Display the resulting frame
            cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Frame", 400, 400)
            cv2.imshow('Frame', frame)

            frame_back = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            canny = cv2.Canny(frame_back,200,400)

            cv2.namedWindow("Canny", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Canny", 400, 400)
            cv2.imshow("Canny", canny)

            # extract orange area
            carImage = extractTheCarArea(frame)

            print(carImage.shape)
            ed.add_new_edges(cv2.Canny(carImage,100,200))

            hsv_green = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            height, width = hsv_green.shape[:2]

            # do template matching
            #match = cv2.imread("bmw.png")
            #frame_copy = frame.copy()
            #match_template(frame, frame_copy, match, 0.7, (0, 255, 0))
            #cv2.imshow("Matched", frame_copy)

            #print hsv_green[height/2][width/2]
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
                
            key = cv2.waitKey(1) & 0xff
            if key == ord('p'):

                while True:

                    key2 = cv2.waitKey(1) or 0xff
                    cv2.imshow('frame', frame)

                    if key2 == ord('p'):
                        break

        # Break the loop
        else:
            break

    
    ed.add_new_edges(cv2.Canny(carImage,100,200), True)
    coords = ed.get_new_scratch()
    h, w = carImage.shape[:2]
    cv2.rectangle(carImage, (int(round(w*coords[1])), int(round(h*coords[0]))), (int(round(w*coords[3])), int(round(h*coords[2]))), (255,0,0), 2)
    
    coords = ed.get_old_scratch()
    h, w = carImage.shape[:2]
    cv2.rectangle(carImage, (int(round(w*coords[1])), int(round(h*coords[0]))), (int(round(w*coords[3])), int(round(h*coords[2]))), (0,0,255), 2)
    
    cv2.imshow("Without tires", carImage)
    cv2.imwrite("result2.jpeg", carImage)
    
    # When everything done, release the video capture object
    #cap.release()

    # Closes all the frames
    #cv2.destroyAllWindows()