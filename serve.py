
from flask import Flask, render_template
from flask_socketio import SocketIO, emit, send
import os
import logging
import logging.handlers
import random
import time
import numpy as np
import skvideo.io
import copy
import cv2
import glob


import utils

import threading

cv2.ocl.setUseOpenCL(False)
random.seed(123)

from pipeline import (
    PipelineRunner,
    ContourDetection,
    Visualizer,
    CsvWriter,
    VehicleCounter)

import datetime
import multiprocessing as mpr
from datetime import datetime
from kalman_filter import KalmanFilter
from tracker import Tracker

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

''' cred = credentials.Certificate('innovahack.json')
default_app = firebase_admin.initialize_app(cred)
db = firestore.client() '''
speed_max = 0
    
IMAGE_DIR = "./out"
VIDEO_SOURCE = "input.mp4"
SHAPE = (720, 1280)  # HxW
EXIT_PTS = np.array([
    [[732, 720], [732, 590], [1280, 500], [1280, 720]],
    [[0, 400], [645, 400], [645, 0], [0, 0]]
])

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)


@app.route("/")
def index():
    ''' thread = threading.Thread(target=main, args=())
    thread.daemon = True
    thread.start() '''
    return render_template("socket.html")


''' @app.route("/demo")
def demo():
    thread = threading.Thread(target=main, args=())
    thread.daemon = True
    thread.start()
    return 'sdfsdf' '''

''' thread = threading.Thread(target=main, args=())
    thread.daemon = True
    thread.start() '''




@socketio.on('connect', namespace='/')
def some_function():
    thread = threading.Thread(target=main, args=())
    thread.daemon = True
    thread.start()
    tracking_cars()


def init_function():
        main("inputone.mp4")
        ''' main("inputtwo.mp4")  '''

def train_bg_subtractor(inst, cap, num=500):
    
    print ('Training BG Subtractor...')
    i = 0
    for frame in cap:
        inst.apply(frame, None, 0.001)
        i += 1
        if i >= num:
            return cap


def main():
    log = logging.getLogger("main")
    
    base = np.zeros(SHAPE + (3,), dtype='uint8')
    exit_mask = cv2.fillPoly(base, EXIT_PTS, (255, 255, 255))[:, :, 0]

    
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500, detectShadows=True)

    
    pipeline = PipelineRunner(pipeline=[
        ContourDetection(bg_subtractor=bg_subtractor,
                         save_image=True, image_dir=IMAGE_DIR),
        
        VehicleCounter(exit_masks=[exit_mask], y_weight=2.0),
        Visualizer(image_dir=IMAGE_DIR),
        CsvWriter(path='./', name='report.csv')
    ], log_level=logging.DEBUG)

    cap = skvideo.io.vreader(VIDEO_SOURCE)

    train_bg_subtractor(bg_subtractor, cap, num=500)

    _frame_number = -1
    frame_number = -1
    for frame in cap:
        if not frame.any():
            log.error("Frame capture failed, stopping...")
            break

        _frame_number += 1

        if _frame_number % 2 != 0:
            continue

        frame_number += 1

        pipeline.set_context({
            'frame': frame,
            'frame_number': frame_number,
        })
        
        pipeline.run()

def tracking_cars():
        the_og_base_url = 'http://wzmedia.dot.ca.gov:1935/D3/89_rampart.stream/'

	BASE_URL = 'http://wzmedia.dot.ca.gov:1935/D3/80_whitmore_grade.stream/'
	FPS = 30
	'''
		Distance to line in road: ~0.025 miles
	'''
	ROAD_DIST_MILES = 0.025

	'''
		Speed limit of urban freeways in California (50-65 MPH)
	'''
	HIGHWAY_SPEED_LIMIT = 65

	# Initial background subtractor and text font
	fgbg = cv2.createBackgroundSubtractorMOG2()
	font = cv2.FONT_HERSHEY_PLAIN

	centers = [] 

	# y-cooridinate for speed detection line
	Y_THRESH = 240

	blob_min_width_far = 6
	blob_min_height_far = 6

	blob_min_width_near = 18
	blob_min_height_near = 18

	frame_start_time = None

	# Create object tracker
	tracker = Tracker(80, 3, 2, 1)

	# Capture livestream
	cap = cv2.VideoCapture ('input.mp4')

	while True:
		centers = []
		frame_start_time = datetime.utcnow()
		ret, frame = cap.read()

		orig_frame = copy.copy(frame)

		#  Draw line used for speed detection
		cv2.line(frame,(0, Y_THRESH),(640, Y_THRESH),(255,0,0),2)


		# Convert frame to grayscale and perform background subtraction
		gray = cv2.cvtColor (frame, cv2.COLOR_BGR2GRAY)
		fgmask = fgbg.apply (gray)

		# Perform some Morphological operations to remove noise
		kernel = np.ones((4,4),np.uint8)
		kernel_dilate = np.ones((5,5),np.uint8)
		opening = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
		dilation = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel_dilate)

		_, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		# Find centers of all detected objects
		for cnt in contours:
			x, y, w, h = cv2.boundingRect(cnt)

			if y > Y_THRESH:
				if w >= blob_min_width_near and h >= blob_min_height_near:
					center = np.array ([[x+w/2], [y+h/2]])
					centers.append(np.round(center))

					cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
			else:
				if w >= blob_min_width_far and h >= blob_min_height_far:
					center = np.array ([[x+w/2], [y+h/2]])
					centers.append(np.round(center))

					cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

		if centers:
			tracker.update(centers)

			for vehicle in tracker.tracks:
				if len(vehicle.trace) > 1:
					for j in range(len(vehicle.trace)-1):
                        # Draw trace line
						x1 = vehicle.trace[j][0][0]
						y1 = vehicle.trace[j][1][0]
						x2 = vehicle.trace[j+1][0][0]
						y2 = vehicle.trace[j+1][1][0]

						cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)

					try:
						'''
							TODO: account for load lag
						'''

						trace_i = len(vehicle.trace) - 1

						trace_x = vehicle.trace[trace_i][0][0]
						trace_y = vehicle.trace[trace_i][1][0]

						# Check if tracked object has reached the speed detection line
						if trace_y <= Y_THRESH + 5 and trace_y >= Y_THRESH - 5 and not vehicle.passed:
							cv2.putText(frame, 'I PASSED!', (int(trace_x), int(trace_y)), font, 1, (0, 255, 255), 1, cv2.LINE_AA)
							vehicle.passed = True

							load_lag = (datetime.utcnow() - frame_start_time).total_seconds()

							time_dur = (datetime.utcnow() - vehicle.start_time).total_seconds() - load_lag
							time_dur /= 60
							time_dur /= 60

							
							vehicle.mph = ROAD_DIST_MILES / time_dur

							# If calculated speed exceeds speed limit, save an image of speeding car
							''' if vehicle.mph > HIGHWAY_SPEED_LIMIT: '''
                                                                
                                                        VehicleCounter.speed_max = int(vehicle.mph)
                                                        ''' VehicleCounter.s = speed_max '''
                                                        ''' VehicleCounter.updateSpeed(speed_max) '''
                                                        print ('UH OH, SPEEDING!')
                                                        cv2.circle(orig_frame, (int(trace_x), int(trace_y)), 20, (0, 0, 255), 2)
                                                        cv2.putText(orig_frame, 'MPH: %s' % int(vehicle.mph), (int(trace_x), int(trace_y)), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
                                                        cv2.imwrite('speeding_%s.png' % vehicle.track_id, orig_frame)
                                                        print ('FILE SAVED!')

					
						if vehicle.passed:
							# Display speed if available
							cv2.putText(frame, 'MPH: %s' % int(vehicle.mph), (int(trace_x), int(trace_y)), font, 1, (0, 255, 255), 1, cv2.LINE_AA)
						else:
							# Otherwise, just show tracking id
							cv2.putText(frame, 'ID: '+ str(vehicle.track_id), (int(trace_x), int(trace_y)), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
					except:
						pass


		# Display all images
		cv2.imshow ('original', frame)
		''' cv2.imshow ('opening/dilation', dilation) '''
		''' cv2.imshow ('background subtraction', fgmask) '''

		# Quit when escape key pressed
		if cv2.waitKey(5) == 27:
			break

		# Sleep to keep video speed consistent
		time.sleep(1.0 / FPS)

	# Clean up
	cap.release()
	cv2.destroyAllWindows()

	# remove all speeding_*.png images created in runtime
	for file in glob.glob('speeding_*.png'):
		os.remove(file)

if __name__ == '__main__':
        ''' print("dfdfdfs") '''
        ''' socketio.run(app) '''
        ''' main() '''
