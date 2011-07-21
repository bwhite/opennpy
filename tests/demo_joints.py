#!/usr/bin/env python
import opennpy
import cv
import frame_convert
import time
import numpy as np
import cPickle as pickle
import os

opennpy.align_depth_to_rgb()
cv.NamedWindow('Video')
print('Press ESC in window to stop')

def get_video():
    return frame_convert.video_cv(opennpy.sync_get_video()[0])

def main():
    out_dir = 'out'
    try:
        os.makedirs(out_dir)
    except OSError:
        pass
    out_dir += '/'
    while 1:
        joints = opennpy.sync_get_joints()
        if not joints:
            vid = opennpy.sync_get_video()[0]
        else:
            vid = joints['image'].copy()
            x, y = (joints['scene'] == 0).nonzero()
            vid[x, y, :] = 0
        vid = frame_convert.video_cv(vid)
        if joints:
            for x in joints['joints'][0].values():
                if x['world'][2] == 0. or x['conf'] < .5:
                    continue
                try:
                    cv.Circle(vid, tuple(map(int, x['image'])), 5, (0, 255, 0), -1)
                except:
                    continue
        cv.ShowImage('Video', vid)
        if joints:
            with open(out_dir + 'kinect-data-%f.pkl' % time.time(), 'w') as fp:
                pickle.dump(joints, fp, -1)
        if cv.WaitKey(10) == 27:
            break
main()
