import numpy as np
import open3d as o3d
import os
import argparse
import sys

parser = argparse.ArgumentParser(description='Visualize point cloud')
parser.add_argument('--file_dir', type=str, help=None)
parser.add_argument('--scene_id', type=str, help=None)
parser.add_argument('--type', type=str, help=None, required=True)
args = parser.parse_args()
print(args)

"""
color_map = [[  0,   0,   0], # unlabeled (white)
             [190, 153, 112], # wall
             [189, 198, 255], # floor
             [213, 255,   0], # cabinet
             [158,   0, 142], # bed
             [152, 255,  82], # chair
             [119,  77,   0], # sofa
             [122,  71, 130], # table
             [  0, 174, 126], # door
             [  0, 125, 181], # window
             [  0, 143, 156], # bookshelf
             [107, 104, 130], # picture
             [255, 229,   2], # counter
             [  1, 255, 254], # desk
             [255, 166, 254], # curtain
             [232,  94, 190], # refridgerator
             [  0, 100,   1], # shower curtain
             [133, 169,   0], # toilet
             [149,   0,  58], # sink
             [187, 136,   0], # bathtub
             [  0,   0, 255]] # otherfurniture (blue)
color_map = np.array(color_map)
"""
s3dis_color = [[255,248,220], [220,220,220], [139,71,38], [238,197,145], [70,130,180], [179,238,58], [110,139,61], [105,105,105], [0,0,128], [205,92,92], [244,164,96], [147,112,219], [255,228,225]]
color_map = np.array(s3dis_color)

def vis(tag):
    file_path = os.path.join(args.file_dir, args.scene_id)
    xyzrgb = np.load(file_path + '_points.npy')
    if tag == 'labels':
        #points = np.load(file_path + '_labels.npy').astype(np.int)
        points = xyzrgb[:, -1].astype(int)
        points = color_map[points, :]
        xyzrgb[:, 3:6] = points
    elif tag == 'preds':
        points = np.load(file_path + '_preds.npy')
        #points = np.argmax(points, axis=1)
        points = color_map[points, :]
        xyzrgb[:, 3:6] = points

    xyzrgb[:, 3:] /= 255
    cache_file = os.path.join(args.file_dir, args.scene_id + '_vis_cached_file.txt')
    np.savetxt(cache_file, xyzrgb)

    pcd = o3d.io.read_point_cloud(cache_file, format='xyzrgb')
    os.remove(cache_file)
    o3d.visualization.draw_geometries([pcd])

if __name__ == '__main__':
    #vis('points')
    #vis('labels')
    #vis('preds')
    vis(args.type)

