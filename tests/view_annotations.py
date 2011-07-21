import glob
import cv
import cPickle as pickle
import time
import frame_convert


def display(joints, depth):
    x, y = (joints['scene'] == 0).nonzero()
    if depth:
        vid = joints['image'].copy()
        vid[x, y, :] = 0
        vid = frame_convert.video_cv(vid)
    else:
        vid = joints['depth'].copy()
        vid[x, y] = 0
    for x in joints['joints'][0].values():
        if x['world'][2] == 0. or x['conf'] < .5:
            continue
        try:
            cv.Circle(vid, tuple(map(int, x['image'])), 5, (0, 255, 0), -1)
        except:
            continue
    cv.ShowImage('Video', vid)


def main(in_dir, depth):
    for x in glob.glob(in_dir + '/*.pkl'):
        with open(x) as fp:
            joints = pickle.load(fp)
        main(joints, depth)
        if cv.WaitKey(10) == 27:
            break
        time.sleep(.1)


def _parse():
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--depth', action='store_true', help='Display depth image')
    
    parser.add_argument('path', type=str, help='Input dir path')
    return parser.parse_args()

args = _parse()
main(args.path, args.depth)
