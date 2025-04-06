import numpy as np
import ctypes as ct
import cv2
import sys
import matplotlib.pyplot as plt
from point2.data_utils.FindGrained_v import FindGrainedLoader
import matplotlib.cm
showsz = 800
mousex, mousey = 0.5, 0.5
zoom = 1.0
changed = True

def onmouse(*args):
    global mousex, mousey, changed
    y = args[1]
    x = args[2]
    mousex = x / float(showsz)
    mousey = y / float(showsz)
    changed = True

cv2.namedWindow('show3d')
cv2.moveWindow('show3d', 800, 800)
cv2.setMouseCallback('show3d', onmouse)

dll = np.ctypeslib.load_library('render_balls_so.dll', '.')

def showpoints(xyz,c_gt=None, c_pred = None, waittime=0, 
    showrot=False, magnifyBlue=0, freezerot=False, background=(0,0,0), 
    normalizecolor=True, ballradius=10):
    global showsz, mousex, mousey, zoom, changed
    xyz=xyz-xyz.mean(axis=0)
    radius=((xyz**2).sum(axis=-1)**0.5).max()
    xyz/=(radius*4.2)/showsz
    if c_gt is None:
        c0 = np.zeros((len(xyz), ), dtype='float32') + 255
        c1 = np.zeros((len(xyz), ), dtype='float32') + 255
        c2 = np.zeros((len(xyz), ), dtype='float32') + 255
    else:
        c0 = c_gt[:, 0]
        c1 = c_gt[:, 1]
        c2 = c_gt[:, 2]


    if normalizecolor:
        c0 /= (c0.max() + 1e-14) / 255.0
        c1 /= (c1.max() + 1e-14) / 255.0
        c2 /= (c2.max() + 1e-14) / 255.0


    c0 = np.require(c0, 'float32', 'C')
    c1 = np.require(c1, 'float32', 'C')
    c2 = np.require(c2, 'float32', 'C')

    show = np.zeros((showsz, showsz, 3), dtype='uint8')
    def render():
        rotmat=np.eye(3)
        if not freezerot:
            xangle=(mousey-0.5)*np.pi*1.2
        else:
            xangle=0
        rotmat = rotmat.dot(
            np.array([
                [1.0, 0.0, 0.0],
                [0.0, np.cos(xangle), -np.sin(xangle)],
                [0.0, np.sin(xangle), np.cos(xangle)],
            ]))
        if not freezerot:
            yangle = (mousex - 0.5) * np.pi * 1.2
        else:
            yangle = 0
        rotmat = rotmat.dot(
            np.array([
                [np.cos(yangle), 0.0, -np.sin(yangle)],
                [0.0, 1.0, 0.0],
                [np.sin(yangle), 0.0, np.cos(yangle)],
            ]))
        rotmat *= zoom
        nxyz = xyz.dot(rotmat) + [showsz / 2, showsz / 2, 0]

        ixyz = nxyz.astype('int32')
        show[:] = background
        dll.render_ball(
            ct.c_int(show.shape[0]), ct.c_int(show.shape[1]),
            show.ctypes.data_as(ct.c_void_p), ct.c_int(ixyz.shape[0]),
            ixyz.ctypes.data_as(ct.c_void_p), c0.ctypes.data_as(ct.c_void_p),
            c1.ctypes.data_as(ct.c_void_p), c2.ctypes.data_as(ct.c_void_p),
            ct.c_int(ballradius))

        if magnifyBlue > 0:
            show[:, :, 0] = np.maximum(show[:, :, 0], np.roll(
                show[:, :, 0], 1, axis=0))
            if magnifyBlue >= 2:
                show[:, :, 0] = np.maximum(show[:, :, 0],
                                           np.roll(show[:, :, 0], -1, axis=0))
            show[:, :, 0] = np.maximum(show[:, :, 0], np.roll(
                show[:, :, 0], 1, axis=1))
            if magnifyBlue >= 2:
                show[:, :, 0] = np.maximum(show[:, :, 0],
                                           np.roll(show[:, :, 0], -1, axis=1))
        if showrot:
            cv2.putText(show, 'xangle %d' % (int(xangle / np.pi * 180)),
                        (30, showsz - 30), 0, 0.5,(255, 0, 0))
            cv2.putText(show, 'yangle %d' % (int(yangle / np.pi * 180)),
                        (30, showsz - 50), 0, 0.5,(255, 0, 0))
            cv2.putText(show, 'zoom %d%%' % (int(zoom * 100)), (30, showsz - 70), 0,
                        0.5, (255, 0, 0))
    changed = True
    while True:
        if changed:
            render()
            changed = False
        cv2.imshow('show3d', show)
        if waittime == 0:
            cmd = cv2.waitKey(10) % 256
        else:
            cmd = cv2.waitKey(waittime) % 256
        if cmd == ord('q'):
            break
        elif cmd == ord('Q'):
            sys.exit(0)

        if cmd == ord('t') or cmd == ord('p'):
            if cmd == ord('t'):
                if c_gt is None:
                    c0 = np.zeros((len(xyz), ), dtype='float32') + 255
                    c1 = np.zeros((len(xyz), ), dtype='float32') + 255
                    c2 = np.zeros((len(xyz), ), dtype='float32') + 255
                else:
                    c0 = c_gt[:, 0]
                    c1 = c_gt[:, 1]
                    c2 = c_gt[:, 2]
            else:
                if c_pred is None:
                    c0 = np.zeros((len(xyz), ), dtype='float32') + 255
                    c1 = np.zeros((len(xyz), ), dtype='float32') + 255
                    c2 = np.zeros((len(xyz), ), dtype='float32') + 255
                else:
                    c0 = c_pred[:, 0]
                    c1 = c_pred[:, 1]
                    c2 = c_pred[:, 2]
            if normalizecolor:
                c0 /= (c0.max() + 1e-14) / 255.0
                c1 /= (c1.max() + 1e-14) / 255.0
                c2 /= (c2.max() + 1e-14) / 255.0
            c0 = np.require(c0, 'float32', 'C')
            c1 = np.require(c1, 'float32', 'C')
            c2 = np.require(c2, 'float32', 'C')
            changed = True

        if cmd==ord('n'):
            zoom*=1.1
            changed=True
        elif cmd==ord('m'):
            zoom/=1.1
            changed=True
        elif cmd==ord('r'):
            zoom=1.0
            changed=True
        elif cmd==ord('s'):
            cv2.imwrite('show3d.png',show)
        if waittime!=0:
            break
    return cmd

if __name__ == '__main__':
    import argparse
    import os
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(BASE_DIR)
    sys.path.append(BASE_DIR)
    sys.path.append(os.path.join(ROOT_DIR, 'data_utils'))
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=True, help='use uniform sampiling')
    parser.add_argument('--data_path', default='Airplane', help='Airplane/Car/Chair')
    args = parser.parse_args()
    data_path = 'D:/localCode/point2/data'
    train_dataset = FindGrainedLoader(root=data_path, args=args, split='test', process_data=False)

    data = train_dataset[111]
    point_set,_ = data

    np.random.seed(100)

    cmap = plt.cm.get_cmap("hsv", 20)
    cmap = np.array([cmap(i) for i in range(20)])[:, :3]
    print(cmap)
    seg = np.ones(2048).astype(int) * 3

    pred_choice = np.ones(1024)
    pred_choice = np.expand_dims(seg, 0).astype(int)
    pred = cmap[pred_choice[0], :]
    gt = cmap[seg, :]

    showpoints(point_set,gt,c_pred=pred, waittime=0, showrot=False, magnifyBlue=0, freezerot=False,
               background=(255, 255, 255), normalizecolor=True, ballradius=10)
#ballradius:size of point
#