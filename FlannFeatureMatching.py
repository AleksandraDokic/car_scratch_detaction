#===============================================================================
# author: Aleksandra Dokic
# data: 25.06.2018
# company: Telemotive AG
#===============================================================================
import cv2
import numpy as np
import os

#===============================================================================
# Class to extract display for Drehmomentschlussel
#===============================================================================
class FindRoi:

    def __init__(self, frame, debug = False):

        self.img = frame
        self.img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.img_matched = frame.copy()
        self.roi = []
        self.roi_min = []
        self.matches = []
        #self.rectange_cutter = DMC_Reader()

        # should use large or small templates for matching? start with small
        self.large = False

        self.template_matching(threshold=0.70, name1='1_g.png', name1s='1_gs.png', name2='2_g.png',name2s='2_gs.png',
                                name3='3_g.png', name3s='3_gs.png', name4='4_g.png',name4s='4_gs.png',match_on=self.img_gray, draw_on=self.img_matched,first_call=True)

        self.roi, self.coordinates = self.find_roi()

        if self.roi != []:
            self.process_roi2()

        if(debug):
            self.show_results()

        '''# in case minimum roi area was not found just reuse the one given by fix cutting of second third or original ROI
        # when roi existed in the first place
        if self.roi != []:
            if self.roi_min == []:
                self.roi_min_thresh = self.value_img_resized
            else:
                self.coordinates = self.coordinates_min'''

    # ===========================================================================
    # Function that cuts roi using found template matches
    #   returns: segmented roi and coordinates
    # ===========================================================================
    def find_roi(self):

        template_combination = self.check_adequate_amount_of_matches()

        # we did not find three valid templates, we can not fix roi
        if template_combination == -1:
            return [], [0, 0, 0, 0, 0, 0, 0, 0]

        #get top left corners of where first three buttons should be
        if template_combination == 1:
            p1 = np.array([self.matches[0][0][0], self.matches[0][0][1]])
            p2 = np.array([self.matches[1][0][0], self.matches[1][0][1]])
            p3 = np.array([self.matches[2][0][0], self.matches[2][0][1]])
        elif template_combination == 2:
            p1 = np.array([self.matches[0][0][0], self.matches[0][0][1]])
            p2 = np.array([self.matches[1][0][0], self.matches[1][0][1]])
            p3 = np.array([self.matches[0][0][0], self.matches[3][0][1]])
        elif template_combination == 3:
            p2 = np.array([self.matches[1][0][0], self.matches[1][0][1]])
            p3 = np.array([self.matches[2][0][0], self.matches[2][0][1]])
            p1 = np.array([self.matches[2][0][0], self.matches[1][0][1]-(self.matches[3][0][1]-self.matches[2][0][1])/2])
        elif template_combination == 4:
            p1 = np.array([self.matches[0][0][0], self.matches[0][0][1]])
            p3 = np.array([self.matches[2][0][0], self.matches[2][0][1]])
            p2 = np.array([self.matches[3][0][0], self.matches[0][0][1]+(self.matches[3][0][1]-self.matches[2][0][1])/2])

        # get the square dimension
        v = p1 - p3
        d = np.sqrt(v[0] * v[0] + v[1] * v[1])

        # get upper line of display vector
        v1 = p1 - p2
        # get it's normal vector
        v2 = np.empty_like(v1)
        v2[0] = -v1[1]
        v2[1] = v1[0]

        # normalize vectors
        v1_norm = np.sqrt(v1[0] * v1[0] + v1[1] * v1[1])
        v1_norm = (v1[0] / v1_norm, v1[1] / v1_norm)
        v2_norm = np.sqrt(v2[0] * v2[0] + v2[1] * v2[1])
        v2_norm = (v2[0] / v2_norm, v2[1] / v2_norm)

        # build up display contour
        contour = []
            # 1. first corner point is from p1 on upper line vector on distance d
        point = [p1[0] + 2.5 * d * v1_norm[0], p1[1] + 2.5 * d * v1_norm[1]]
        contour.append(point)
            # 2. second corner point is p1
        contour.append(p1)
            # 3. third corner point is from p1 on normal vector on distance d
        point = [p1[0] - 2 * d * v2_norm[0], p1[1] - 2 * d * v2_norm[1]]
        contour.append(point)
            # 4. fourth corner point is from second one on normal vector on distance d
        point = [contour[0][0] - 2 * d * v2_norm[0], contour[0][1] - 2 * d * v2_norm[1]]
        contour.append(point)
        contour = np.array(contour)

        # draw detected contour
        cv2.polylines(self.img_matched, np.int32([contour]), True, (0, 255, 255),2)
        print(contour)
        return self.rectange_cutter.fix_rotation_and_cut_dmc(self.img_gray, contour, debug=False)

    # ===========================================================================
    # Function that checks if we at least have three valid template matches
    # and returns information which of template combinations can be used
    #   --------------------   -----    -----
    #   |                  |   | 1 |    | 2 |
    #   |      display     |   -----    -----
    #   |                  |   -----    -----
    #   |                  |   | 3 |    | 4 |
    #   --------------------   -----    -----
    #   returns:
    #       no three valid templates found: -1
    #       templates 1, 2 and 3 are used: 1
    #       templates 1, 2 and 4 are used: 2
    #       templates 2, 3 and 4 are used: 3
    #       templates 1, 3 and 4 are used: 4
    # ===========================================================================
    def check_adequate_amount_of_matches(self):

        firstValid = True
        if self.matches[0][0] != -1 and self.matches[1][0] != -1 and self.matches[2][0] != -1:
            ok_possition1 = self.matches[0][0][0] < (self.matches[1][0][0] - abs(self.matches[0][0][0]-self.matches[0][1][0])/2)
            ok_possition2 = self.matches[0][0][1] < (self.matches[2][0][1]-abs(self.matches[0][0][1]-self.matches[0][1][1])/2)
            if ok_possition1 and ok_possition2:
                return 1
            else:
                firstValid = False

        if firstValid and self.matches[0][0] != -1 and self.matches[1][0] != -1 and self.matches[3][0] != -1:
            ok_possition = self.matches[0][0][1] < (
                        self.matches[3][0][1] - abs(self.matches[0][0][1] - self.matches[0][1][1]) / 2)
            if ok_possition:
                return 2
            else:
                return -1

        if self.matches[3][0] != -1 and self.matches[1][0] != -1 and self.matches[2][0] != -1:
            return 3

        if self.matches[0][0] != -1 and self.matches[2][0] != -1 and self.matches[3][0] != -1:
            ok_possition1 = self.matches[0][0][0] < (
                        self.matches[3][0][0] - abs(self.matches[0][0][0] - self.matches[0][1][0]) / 2)
            ok_possition2 = self.matches[0][0][1] < (
                        self.matches[2][0][1] - abs(self.matches[0][0][1] - self.matches[0][1][1]) / 2)
            if ok_possition1 and ok_possition2:
                return 4

        return -1

    def process_roi(self):

        self.histogram = cv2.equalizeHist(self.roi)

        blurred_img = cv2.medianBlur(self.histogram, 5)

        ret, self.thresholded_image = cv2.threshold(blurred_img, 230, 255, cv2.THRESH_BINARY)

        kernel5x5 = np.ones((5, 5), np.uint8)
        dilate = cv2.dilate(self.thresholded_image, kernel5x5, iterations=2)
        erode = cv2.erode(dilate, kernel5x5, iterations=2)

        self.erroded_image = erode

        cv2.imshow("Erroded image", self.erroded_image)

        #roi_height, roi_width = self.roi.shape
        #upper_third = self.erroded_image[0:0+roi_height/3, 0:roi_width]
        #second_third = self.erroded_image[roi_height/3:roi_width*2/3, 0:roi_width]

        #self.letter_information = self.extract_biggest_contour(upper_third)
        #self.value = self.extract_biggest_contour(second_third)

        #cv2.imshow("First third", upper_third)
        #cv2.imshow("Second third", second_third)

    def process_roi2(self):

        roi_height, roi_width = self.roi.shape

        self.value_img = self.roi[int(roi_height/3):int(roi_height*2/3), 0:roi_width]

        # resize to pixel height 100
        hightDst = 100.0
        shape = self.value_img.shape
        hightSmall = shape[0]
        factor = (hightDst / hightSmall)
        self.value_img_resized = cv2.resize(self.value_img, (0, 0), fx=factor, fy=factor)

        # do histogram equalization, threshold the image for high values and blur it out
        self.value_img_hist = cv2.equalizeHist(self.value_img_resized)
        ret, self.value_img_tresh = cv2.threshold(self.value_img_hist, 230, 255, cv2.THRESH_BINARY)
        '''self.value_img_blur = cv2.medianBlur(self.value_img_tresh, 7)

        # find contours in this threshholded blurred image, merge two biggest contours which should be numbers
        # extend the region a bit and cut it out as new ROI, if there are not at least two contours keep the
        # previous bigger ROI
        im2, contours, hierarchy = cv2.findContours(self.value_img_blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

        if len(cntsSorted) > 1:

            points = []
            self.cnt_img = cv2.cvtColor(self.value_img_resized, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(self.cnt_img, cntsSorted, 0, (0, 255, 0), 3)
            cv2.drawContours(self.cnt_img, cntsSorted, 1, (255, 0, 0), 3)

            for i in cntsSorted[0]:
                points.append(i)
            for i in cntsSorted[1]:
                points.append(i)
            points = np.asarray(points)

            rect = cv2.minAreaRect(points)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(self.cnt_img, [box], 0, (0, 0, 255), 2)

            # extend area by 35% to be sure to catch all the digit even if they are not part of merged contour
            self.roi_min, self.coordinates_min = self.rectange_cutter.fix_rotation_and_cut_dmc(self.value_img_resized, box, debug=False, percantage=0.35)

            # now cut out region apply histogram and treshhold and use this image for tesseract
            self.roi_min_hist = cv2.equalizeHist(self.roi_min)
            ret, self.roi_min_thresh = cv2.threshold(self.roi_min_hist, 200, 255, cv2.THRESH_BINARY_INV)'''

    def laplacian_transform(self, img):

        (height, width) = img.shape
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        laplacianBlur = cv2.GaussianBlur(laplacian, (9, 9), 0)
        minLaplaceVal = laplacianBlur.min()
        minThres = (minLaplaceVal / 15.0)

        imgLaplacianBinarized = np.ones((height, width), np.uint8) * 255

        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if (laplacianBlur.item(y, x) < minThres):
                    imgLaplacianBinarized.itemset((y, x), 0)

        return imgLaplacianBinarized

    def remove_line(self, img):

        minLineLength = 200
        maxLineGap = 20
        inverted = (255-img)
        lines = cv2.HoughLinesP(inverted, 1, np.pi / 180, 100, minLineLength, maxLineGap)

        color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        copy_img = img.copy()

        if lines is None:
            return copy_img, color_img

        for x1, y1, x2, y2 in lines[0]:

            p1 = np.array([x1, y1])
            p2 = np.array([x2, y2])
            v = p1 - p2

            v_norm = np.sqrt(v[0] * v[0] + v[1] * v[1])
            v_norm = (v[0] / v_norm, v[1] / v_norm)

            d = 10

            p2[0] -= (p2[0]*2/3) * v_norm[0]
            p2[1] -= (p2[0]*2/3) * v_norm[1] - 5
            p1[0] += (p1[0]*2/3) * v_norm[0]
            p1[1] += (p1[0]*2/3) * v_norm[1] + 5

            cv2.line(color_img, (p1[0], p1[1]), (p2[0], p2[1]), (0, 255, 0), 8)
            cv2.line(copy_img, (p1[0], p1[1]), (p2[0], p2[1]), (255, 255, 255), 10)

            cv2.line(color_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

        return copy_img, color_img

    # ===========================================================================
    # Function that matches adequate templates in the input image
    # Depending on class variable self.large we will try or to match an image to
    # larger templates or smaller templates, if there are not at least three
    # acceptable matches found we will replace the template set used and try again
    # After each call self.large is remembered so he will always try to match next
    # frame to the templates that were proven useful in previous provided frame
    #   param: threshold - how close match of template has to be to be accepted
    #   param: name1..4 - template img names for larger template version
    #   param: name1s..4s - template img name for smaller template version
    #   param: match_on - frame img where matches should be located
    #   param: draw_on - img where we want matches drawn on
    #   param: first_call - if this is first recursive call of function
    #   returns: nothing
    # ===========================================================================
    def template_matching(self, threshold, name1, name1s, name2, name2s, name3, name3s, name4, name4s, match_on, draw_on, first_call):

        # go to folder where templates are located
        owd = os.getcwd()
        os.chdir(owd + "\\templates")

        # Read the templates
        if(self.large):
            m1 = cv2.imread(name1, 0)
            m2 = cv2.imread(name2, 0)
            m3 = cv2.imread(name3, 0)
            m4 = cv2.imread(name4, 0)
        else:
            m1 = cv2.imread(name1s, 0)
            m2 = cv2.imread(name2s, 0)
            m3 = cv2.imread(name3s, 0)
            m4 = cv2.imread(name4s, 0)

        # return to the valid project directory
            os.chdir(owd)

        self.match_template(match_on, draw_on, m1, threshold, (0, 255, 0))
        self.match_template(match_on, draw_on, m2, threshold, (0, 0, 255))
        self.match_template(match_on, draw_on, m3, threshold, (255, 0, 0))
        self.match_template(match_on, draw_on, m4, threshold, (255, 255, 0))

        if(first_call and self.check_adequate_amount_of_matches() == -1):
            self.matches = []
            self.large = not self.large
            self.template_matching(threshold, name1, name1s, name2, name2s, name3, name3s, name4, name4s, match_on, draw_on, not first_call)

        return

    # ===========================================================================
    # Function that matches one given template in given image according to threshold
    # and draws it in given color
    #   param: img - original image where to match template to
    #   param: template - template to be matched
    #   param: threshold - how close match of template has to be to be accepted
    #   param: color - matched template will be draw in this color on self.img_matched
    #   returns: if the template with given threshold has been found in provided frame
    # ===========================================================================
    def match_template(self, img_src, img_dst, template, threshold, color):

        # Store width and heigth of template in w and h
        w, h = template.shape[::-1]

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
            self.matches.append([top_left, bottom_right])
            return True
        self.matches.append([-1, -1])
        return False

    # ===========================================================================
    # Utility functions to show results for debug
    # ===========================================================================
    def show_results(self):
        cv2.imshow("Original", self.img)
        cv2.imshow("Template matching", self.img_matched)

        if self.roi != []:
            cv2.imshow("Roi", self.roi)

            cv2.imshow("Value", self.value_img_resized)
            cv2.imshow("Value hist", self.value_img_hist)
            cv2.imshow("Value thresh", self.value_img_tresh)

            #cv2.imshow("Value blured", self.value_img_blur)

            '''if self.roi_min != []:
                cv2.imshow("Contours", self.cnt_img)
                cv2.imshow("Roi min", self.roi_min)
                cv2.imshow("Roi min thresh", self.roi_min_thresh)'''


def extractBlueAreas2(img):
    # segment the biggest blue region inside an image and crop it out for tesseract recognition
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

    roi = img.copy()

    if len(contours) != 0:
        # draw in blue the contours that were founded
        cv2.drawContours(img_copy, contours, -1, 255, 3)

        # find the biggest area
        c = max(contours, key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(c)
        roi = roi[y:y + h][x:x+w]
        # draw the book contour (in green)
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
    else:
        print("No cntours")

    # show the images
    # cv2.imshow("Result",roi)

    #self.imgExtractBlue = self.imgExtractBlueThreshold  # dummy-assignement to avoid crash on imshow!
    #self.imgExtractBlueThresholdOpened = self.fillSmallGaps(self.imgExtractBlueThreshold)


def match_template(img_src, img_dst, template, threshold, color):

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
        return False

def define_random_point(img):
    pass

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture('demaged.mp4')

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

        canny = cv2.Canny(frame_back,100,200)

        cv2.namedWindow("Canny", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Canny", 400, 400)
        cv2.imshow("Canny", canny)

        # extract orange area
        extractBlueAreas2(frame)

        hsv_green = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        height, width = hsv_green.shape[:2]

        # do template matching

        match = cv2.imread("bmw.png")
        frame_copy = frame.copy()
        #match_template(frame, frame_copy, match, 0.7, (0, 255, 0))

        cv2.imshow("Matched", frame_copy)

        crop_img = frame[height/2:height/2 + height/10, width/2:width/2 + width/10]

        cv2.imshow("cropped", crop_img)


        print hsv_green[height/2][width/2]
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