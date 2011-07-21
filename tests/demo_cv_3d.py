#!/usr/bin/env python
import opennpy
import cv
import frame_convert
import matplotlib.pyplot as mp
import random
from mpl_toolkits.mplot3d import Axes3D

cv.NamedWindow('Depth')
print('Press ESC in window to stop')

mp.ion()
fig = mp.figure()
def get_depth():
    return frame_convert.pretty_depth_cv(opennpy.sync_get_depth()[0])

while 1:
    #mp.clf()
    #mp.hist(opennpy.sync_get_depth()[0].ravel(), 500)
    #mp.draw()
    #continue
    pts = [x for x in opennpy.depth_to_3d(opennpy.sync_get_depth()[0]).reshape((640*480, 3)) if x[2] != 0 and x[2] < 2000]
    try:
        x, y, z = zip(*random.sample(pts, min(5000, len(pts))))
    except ValueError:
        continue
    mp.clf()
    Axes3D(fig).scatter(x, y, z)
    mp.draw()
    #cv.ShowImage('Depth', )
    #if cv.WaitKey(10) == 27:
    #    break
