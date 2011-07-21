import glob
import cv
import cPickle as pickle
import time
import frame_convert
in_dir = 'out/brandyn'


def main(joints):
    vid = joints['image'].copy()
    x, y = (joints['scene'] == 0).nonzero()
    vid[x, y, :] = 0
    vid = frame_convert.video_cv(vid)
    for x in joints['joints'][0].values():
        if x['world'][2] == 0. or x['conf'] < .5:
            continue
        try:
            cv.Circle(vid, tuple(map(int, x['image'])), 5, (0, 255, 0), -1)
        except:
            continue
    cv.ShowImage('Video', vid)

for x in glob.glob(in_dir + '/*.pkl'):
    with open(x) as fp:
        joints = pickle.load(x)
    main(joints)
    if cv.WaitKey(10) == 27:
        break
    time.sleep(.1)

