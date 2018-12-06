from flask import Flask, render_template
from flask_socketio import SocketIO, emit, send
''' from flask_script import Manager '''

import os
import logging
import logging.handlers
import random

import numpy as np
import skvideo.io
import cv2
import matplotlib.pyplot as plt

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
def hello():
    return render_template("socket.html")

@socketio.on('connect', namespace='/')
def some_function():
    ''' emit('message', VehicleCounter.counter) '''
    emit('message', "dddd")



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


if __name__ == '__main__':
    ''' log = utils.init_logging() '''
    ''' log.debug("Creating image directory `%s`...", IMAGE_DIR) '''
    ''' manager.add_command('hello', initial_main())
    manager.run({ 'hello': initial_main() }) '''
    socketio.run(app)
    ''' manager.run(socketio.run(app)) '''
