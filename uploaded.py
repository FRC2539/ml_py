import argparse

import cv2
import numpy as np
from time import time
import tflite_runtime.interpreter as tflite
from cscore import CameraServer, VideoSource, UsbCamera, MjpegServer
from networktables import NetworkTablesInstance
import cv2
import collections
import json
import sys

class ConfigParser:
    def __init__(self, config_path):
        self.team = -1

        # parse file
        try:
            with open(config_path, "rt", encoding="utf-8") as f:
                j = json.load(f)
                print(str(j))
        except OSError as err:
            print("could not open '{}': {}".format(config_path, err), file=sys.stderr)

        # top level must be an object
        if not isinstance(j, dict):
            self.parseError("must be JSON object", config_path)

        # team number
        try:
            self.team = j["team"]
        except KeyError:
            self.parseError("could not read team number", config_path)
        
        # # red
        # try:
        #     self.red = j["red"]
        # except KeyError:
        #     self.parseError("could not read red", config_path)    

        # # redtolerance
        # try:
        #     self.redtolerance = j["redtolerance"]
        # except KeyError:
        #     self.parseError("could not read red tolerance", config_path)    


        # # blue
        # try:
        #     self.red = j["blue"]
        # except KeyError:
        #     self.parseError("could not read blue", config_path)   
        
        # # bluetolerance
        # try:
        #     self.redtolerance = j["bluetolerance"]
        # except KeyError:
        #     self.parseError("could not read blue tolerance", config_path)  

        # cameras
        try:
            self.cameras = j["cameras"]
        except KeyError:
            self.parseError("could not read cameras", config_path)

    def parseError(self, str, config_file):
        """Report parse error."""
        print("config error in '" + config_file + "': " + str, file=sys.stderr)


class PBTXTParser:
    def __init__(self, path):
        self.path = path
        self.file = None

    def parse(self):
        with open(self.path, 'r') as f:
            self.file = ''.join([i.replace('item', '') for i in f.readlines()])
            blocks = []
            obj = ""
            for i in self.file:
                if i == '}':
                    obj += i
                    blocks.append(obj)
                    obj = ""
                else:
                    obj += i
            self.file = blocks
            label_map = []
            for obj in self.file:
                obj = [i for i in obj.split('\n') if i]
                name = obj[2].split()[1][1:-1]
                label_map.append(name)
            self.file = label_map

    def get_labels(self):
        return self.file


class BBox(collections.namedtuple('BBox', ['xmin', 'ymin', 'xmax', 'ymax'])):
    """Bounding box.
    Represents a rectangle which sides are either vertical or horizontal, parallel
    to the x or y axis.
    """
    __slots__ = ()

    def scale(self, sx, sy):
        """Returns scaled bounding box."""
        return BBox(xmin=(sx * self.xmin),
                    ymin=(sy * self.ymin),
                    xmax=(sx * self.xmax),
                    ymax=(sy * self.ymax))


class Tester:
    def __init__(self, config_parser):
        print("Initializing TFLite runtime interpreter")
        try:
            model_path = "model.tflite"
            self.interpreter = tflite.Interpreter(model_path, experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
            self.hardware_type = "Coral Edge TPU"
        except:
            print("Failed to create Interpreter with Coral, switching to unoptimized")
            model_path = "unoptimized.tflite"
            self.interpreter = tflite.Interpreter(model_path)
            self.hardware_type = "Unoptimized"

        self.interpreter.allocate_tensors()

        print("Getting labels")
        parser = PBTXTParser("map.pbtxt")
        parser.parse()
        #self.labels = parser.get_labels()
        self.labels = ['red', 'blue', 'invalid']

        print("Connecting to Network Tables")
        ntinst = NetworkTablesInstance.getDefault()
        ntinst.startClientTeam(config_parser.team)
        ntinst.startDSClient()
        
        self.entry_filterColor = ntinst.getTable("ML").getEntry("filterColor")
        self.entry_targetAcquired = ntinst.getTable("ML").getEntry("targetAcquired")
        self.entry = ntinst.getTable("ML").getEntry("detections")
        self.entry_targetX = ntinst.getTable("ML").getEntry("targetX")
        self.entry_targetY = ntinst.getTable("ML").getEntry("targetY")
        self.entry_targetArea = ntinst.getTable("ML").getEntry("targetArea")
        self.entry_targetColor = ntinst.getTable("ML").getEntry("targetColor")

        self.calibrate = ntinst.getTable("ML").getEntry("calibrate")
        self.calibrateFound = ntinst.getTable("ML").getEntry("calibrateFound")
        self.coral_entry = ntinst.getTable("ML").getEntry("coral")
        self.fps_entry = ntinst.getTable("ML").getEntry("fps")
        #self.resolution_entry = ntinst.getTable("ML").getEntry("resolution")
        self.resolution_entryX = ntinst.getTable("ML").getEntry("resolutionX")
        self.resolution_entryY = ntinst.getTable("ML").getEntry("resolutionY")
        self.feed = ntinst.getTable("ML").getEntry("feed")
        self.mldisable = ntinst.getTable("ML").getEntry("mldisable")
        self.temp_detectedBalls = []

        print("Starting camera server")
        cs = CameraServer.getInstance()
        camera = cs.startAutomaticCapture()
        camera_config = config_parser.cameras[0]
        WIDTH, HEIGHT = camera_config["width"], camera_config["height"]
        camera.setResolution(WIDTH, HEIGHT)
        self.cvSink = cs.getVideo()
        self.img = np.zeros(shape=(HEIGHT, WIDTH, 3), dtype=np.uint8)
        self.output = cs.putVideo("Axon", WIDTH, HEIGHT)
        self.frames = 0

        self.coral_entry.setString(self.hardware_type)
        self.resolution_entryX.setNumber(WIDTH)
        self.resolution_entryY.setNumber(HEIGHT)
        self.feed.setString("http://wpilibpi.local:1182/stream.mjpg")
        self.entry_targetAcquired.setBoolean(0)
        self.entry_filterColor.setString("Init_ML")
        #self.entry_targetColor.setString("red")
        self.calibrate.setBoolean(0)
        self.calibrateFound.setString("")

    def isWithinTolerance(self, arr1, arr2, tolerance):
        for i in range(len(arr1)):
            if abs(arr1[i] - arr2[i]) > tolerance[i]:
                return False
        return True

    def run(self):
        print("mldisable: "+str(self.mldisable))

        
        while True and str(self.mldisable) !== "true":
            start = time()
            # Acquire frame and resize to expected shape [1xHxWx3]
            ret, frame_cv2 = self.cvSink.grabFrame(self.img)
            if not ret:
                #print("Image failed")
                continue

            # input
            scale = self.set_input(frame_cv2)

            # run inference
            self.interpreter.invoke()
            #foundBalls = 0
            # output
            boxes, class_ids, scores, x_scale, y_scale = self.get_output(scale)

            for i in range(len(boxes)):
                if scores[i] > 0.5:
                    
                    ymin, xmin, ymax, xmax = boxes[i]

                    bbox = BBox(xmin=xmin,
                                ymin=ymin,
                                xmax=xmax,
                                ymax=ymax).scale(x_scale, y_scale)
                    ymin, xmin, ymax, xmax = int(bbox.ymin), int(bbox.xmin), int(bbox.ymax), int(bbox.xmax)

                    zoomAmount = 0.55
                    width = xmax - xmin
                    height = ymax - ymin
                    xmin = int(xmin + width * (zoomAmount * 0.5))
                    ymin = int(ymin + height * (zoomAmount * 0.5))
                    xmax = int(xmax - width * (zoomAmount * 0.5))
                    ymax = int(ymax - height * (zoomAmount * 0.5))

                    height, width, channels = frame_cv2.shape

                    if not 0 <= ymin < ymax <= height or not 0 <= xmin < xmax <= width:
                        print('invalid')
                        print(xmin, xmax, ymin, ymax)
                        continue

                    #[ 30.55803571  46.90922619 200.59970238][ 40.93589744 229.6474359  251.3974359 ]
                    #[172.40277778 126.6558642   59.11419753]

                    red = [20, 35, 190]
                    redtolerance = [50, 50, 50]
                    blue = [150, 115, 49]
                    bluetolerance = [50, 50, 50]
                    hilight = [15, 55, 255] #[0, 255, 255] yellow
                    text = [15, 55, 255]

                    cropped = frame_cv2[ymin:ymax, xmin: xmax]
                    averages = np.average(cropped, axis=(0, 1))

                    if self.isWithinTolerance(red, averages, redtolerance):
                        #foundBalls += 1
                        class_ids[i] = 0
                        cv2.rectangle(frame_cv2, (xmin, ymin), (xmax, ymax), red, 2)
                        frame_cv2 = self.label_frame(frame_cv2, "red", boxes[i], scores[i], x_scale, y_scale, averages)
                        #frame_cv2 = self.label_frame(frame_cv2, "Red: "+str(averages), boxes[i], scores[i], x_scale, y_scale)
                    elif self.isWithinTolerance(blue, averages, bluetolerance):
                        #foundBalls += 1
                        class_ids[i] = 1
                        cv2.rectangle(frame_cv2, (xmin, ymin), (xmax, ymax), blue, 2)
                        frame_cv2 = self.label_frame(frame_cv2, "blue", boxes[i], scores[i], x_scale, y_scale, averages)
                        #frame_cv2 = self.label_frame(frame_cv2, "Blue: "+str(averages), boxes[i], scores[i], x_scale, y_scale)
                    else:
                        if self.calibrate.getBoolean(0) == 1:
                            class_ids[i] = 2
                            cv2.rectangle(frame_cv2, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                            frame_cv2 = self.label_frame(frame_cv2, str("not red or blue"), boxes[i], scores[i], x_scale, y_scale, averages)
                            print("not red or blue: "+str(averages))
                        pass
            
            if self.calibrate.getBoolean(0) == 1:
                self.calibrateFound.setString(str(self.temp_detectedBalls))
            
            #filterKey = ""

            filterKey = self.entry_filterColor.getString("None")
            #print("have a filter: "+str(filterKey))
            if len(self.temp_detectedBalls):

                self.temp_detectedBalls.sort(reverse=True, key=lambda e: e['area'])
                if (filterKey == 'red' or filterKey == 'blue'):
                    self.temp_detectedBalls = list(filter(lambda elem : elem['color'] == filterKey, self.temp_detectedBalls))
                    #print("have a filter: "+str(filterKey))
                else:
                    #print("no filter"+str(filterKey))
                    pass

                #closestBall = self.temp_detectedBalls[0]
                #print("filtered: "+str(self.temp_detectedBalls))
                if len(self.temp_detectedBalls):
                    self.entry_targetAcquired.setBoolean(1)
                    self.entry_targetColor.setString(self.temp_detectedBalls[0]['color'])
                    self.entry_targetX.setNumber(self.temp_detectedBalls[0]['x'])
                    self.entry_targetY.setNumber(self.temp_detectedBalls[0]['y'])
                    self.entry_targetArea.setNumber(self.temp_detectedBalls[0]['area'])
                    cv2.rectangle(frame_cv2, (self.temp_detectedBalls[0]['xmin'], self.temp_detectedBalls[0]['ymin']), (self.temp_detectedBalls[0]['xmax'], self.temp_detectedBalls[0]['ymax']), hilight, 6)
                    frame_cv2 = self.label_frame(frame_cv2, "Target", boxes[i], scores[i], x_scale, y_scale, averages)
                else:
                    self.entry_targetAcquired.setBoolean(0)
            else:
                self.entry_targetAcquired.setBoolean(0)


            cv2.putText(frame_cv2, "fps: " + str(round(1 / (time() - start))) + " found: "+str(len(self.temp_detectedBalls))+" filter:"+str(filterKey), (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, [15, 55, 255], 2)
            self.output.putFrame(frame_cv2)
            
            self.temp_detectedBalls = []

            if self.frames % 100 == 0:
                print("Completed", self.frames, "frames. FPS:", (1 / (time() - start)))
            if self.frames % 10 == 0:
                self.fps_entry.setNumber((1 / (time() - start)))
            self.frames += 1

    

    def label_frame(self, frame, object_name, box, score, x_scale, y_scale, averages):
        #print("box x:"+str(x_scale)+" y:"+str(y_scale))
        ymin, xmin, ymax, xmax = box
        score = float(score)
        bbox = BBox(xmin=xmin,
                    ymin=ymin,
                    xmax=xmax,
                    ymax=ymax).scale(x_scale, y_scale)

        height, width, channels = frame.shape
        # check bbox validity
        if not 0 <= ymin < ymax <= height or not 0 <= xmin < xmax <= width:
            #print("bbox valid: "+str(0 <= ymin < ymax <= height)+ "2: " +str(0 <= xmin < xmax <= width))
            return frame
        #else:
            #print("bbox not valid ymin: "+str(ymin)+ " ymax: " +str(ymax) + " height: "+ str(height))
        #print("got it")
        

        ymin, xmin, ymax, xmax = int(bbox.ymin), int(bbox.xmin), int(bbox.ymax), int(bbox.xmax)
        #self.temp_entry.append({"label": object_name, "box": {"ymin": ymin, "xmin": xmin, "ymax": ymax, "xmax": xmax},
        #                        "confidence": score})

        #add items
        theX = ((xmax-xmin)/2)+xmin
        theY = ((ymax-ymin)/2)+ymin
        theArea = (((xmax-xmin)*(ymax-ymin)))
        self.temp_detectedBalls.append({'x':theX, 'y':theY,'area':theArea,'color':object_name,'xmin':xmin,'xmax':xmax,'ymin':ymin,'ymax':ymax, 'averages':averages});

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 4)

        # Draw label
        # Look up object name from "labels" array using class index
        label = '%s: %d%%' % (object_name, round(score * 100))  # Example: 'person: 72%'
        label_size, base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)  # Get font size
        label_ymin = max(ymin, label_size[1] + 10)  # Make sure not to draw label too close to top of window
        cv2.rectangle(frame, (xmin, label_ymin - label_size[1] - 10), (xmin + label_size[0], label_ymin + base - 10),
                      (255, 255, 255), cv2.FILLED)
        # Draw label text
        cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        #print("should see text")
        return frame

    def input_size(self):
        """Returns input image size as (width, height) tuple."""
        _, height, width, _ = self.interpreter.get_input_details()[0]['shape']
        return width, height

    def set_input(self, frame):
        """Copies a resized and properly zero-padded image to the input tensor.
        Args:
          frame: image
        Returns:
          Actual resize ratio, which should be passed to `get_output` function.
        """
        width, height = self.input_size()
        h, w, _ = frame.shape
        new_img = np.reshape(cv2.resize(frame, (300, 300)), (1, 300, 300, 3))
        self.interpreter.set_tensor(self.interpreter.get_input_details()[0]['index'], np.copy(new_img))
        return width / w, height / h

    def output_tensor(self, i):
        """Returns output tensor view."""
        tensor = self.interpreter.get_tensor(self.interpreter.get_output_details()[i]['index'])
        return np.squeeze(tensor)

    def get_output(self, scale):
        boxes = self.output_tensor(0)
        class_ids = self.output_tensor(1)
        scores = self.output_tensor(2)

        width, height = self.input_size()
        image_scale_x, image_scale_y = scale
        x_scale, y_scale = width / image_scale_x, height / image_scale_y
        return boxes, class_ids, scores, x_scale, y_scale


if __name__ == '__main__':
    config_file = "/boot/frc.json"
    config_parser = ConfigParser(config_file)
    tester = Tester(config_parser)
    tester.run()
