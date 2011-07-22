import glob
import cv
import cPickle as pickle
import time
import frame_convert
import distpy
import numpy as np
import opennpy


colors = np.random.random((25, 3)) * 255
colors[0, :] = 0
colors = np.array(colors, dtype=np.uint8)


def display(joints, display_type, prune_joints):
    scene_x, scene_y = (joints['scene'] == 0).nonzero()
    # If any joints are in the background then skip this annotation
    if prune_joints:
        for player_joints in joints['joints'].values():
            image_joints = [(y, x['image']) for y, x in player_joints.items()
                            if x['conf'] >= .5 and x['world'][2] != 0.]
            for joint_name, image_joint in image_joints:
                image_joint_x, image_joint_y = np.array(image_joint, dtype=np.int)
                print(joint_name)
                print(joints['scene'][image_joint_y, image_joint_x])
                try:
                    if joints['scene'][image_joint_y, image_joint_x] == 0:
                        return
                except IndexError:
                    return
    if display_type == 'depth':
        vid = joints['depth'].copy()
        vid[scene_x, scene_y] = 0
        vid = frame_convert.pretty_depth_cv(vid)
    elif display_type == 'image':
        vid = joints['image'].copy()
        vid[scene_x, scene_y, :] = 0
        vid = frame_convert.video_cv(vid)
    elif display_type == 'joints':
        #assert('depth_registered' in joints and not joints['depth_registered'])
        vid = joints['depth'].copy()
        vid[scene_x, scene_y] = 0
        out = np.zeros((480, 640), dtype=np.uint8)
        #pts = np.asfarray([[j, i, vid[i, j]] for i in range(480) for j in range(640)]).reshape((480, 640))
        #pts = np.asfarray([[j, i] for i in range(480) for j in range(640)]).reshape((480, 640, 2))
        pts = opennpy.depth_to_3d(vid)
        dist = distpy.L2Sqr()
        neighbors = np.array([x['image'] for x in joints['joints'][0].values()
                              if x['conf'] >= .5 and x['world'][2] != 0.])
        for y, x in zip(*vid.nonzero()):
            out[y, x] = dist.nn(neighbors, pts[y, x])[1] + 1
        vid = colors[out.ravel()].reshape((480, 640, 3))
        vid = frame_convert.video_cv(vid)
    for x in joints['joints'][0].values():
        if x['world'][2] == 0. or x['conf'] < .5:
            continue
        try:
            cv.Circle(vid, tuple(map(int, x['image'])), 5, (0, 255, 0), -1)
        except:
            continue
    return vid


def main(in_dir, display_type, prune_joints):
    for x in glob.glob(in_dir + '/*.pkl'):
        with open(x) as fp:
            joints = pickle.load(fp)
        vid = display(joints, display_type, prune_joints)
        if not vid:
            continue
        cv.ShowImage('Video', vid)
        if cv.WaitKey(10) == 27:
            break
        time.sleep(.1)


def _parse():
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--type', default='image', choices=['image', 'depth', 'joints'], help='Display type')
    parser.add_argument('--prune_joints', action='store_true', help='Display type')
    parser.add_argument('path', help='Input dir path')
    return parser.parse_args()

args = _parse()
main(args.path, args.type, args.prune_joints)
