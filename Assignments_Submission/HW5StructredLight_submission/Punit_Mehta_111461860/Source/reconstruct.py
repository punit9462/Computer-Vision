# ================================================
# Skeleton codes for HW5
# Read the skeleton codes carefully and put all your
# codes into function "reconstruct_from_binary_patterns"
# ================================================

import cv2
import numpy as np
from math import log, ceil, floor
import matplotlib.pyplot as plt
import pickle
import sys

def help_message():
    # Note: it is assumed that "binary_codes_ids_codebook.pckl", "stereo_calibration.pckl",
    # and images folder are in the same root folder as your "generate_data.py" source file.
    # Same folder structure will be used when we test your program

    print("Usage: [Output_Directory]")
    print("[Output_Directory]")
    print("Where to put your output.xyz")
    print("Example usages:")
    print(sys.argv[0] + " ./")


def reconstruct_from_binary_patterns():
    scale_factor = 1.0
    ref_white = cv2.resize(cv2.imread("images/pattern000.jpg", cv2.IMREAD_GRAYSCALE) / 255.0, (0, 0), fx=scale_factor,
                           fy=scale_factor)
    ref_black = cv2.resize(cv2.imread("images/pattern001.jpg", cv2.IMREAD_GRAYSCALE) / 255.0, (0, 0), fx=scale_factor,
                           fy=scale_factor)
    ref_avg = (ref_white + ref_black) / 2.0

    ref_on = ref_avg + 0.05  # a threshold for ON pixels
    ref_off = ref_avg - 0.05  # add a small buffer region

    h, w = ref_white.shape

    # mask of pixels where there is projection
    proj_mask = (ref_white > (ref_black + 0.05))

    scan_bits = np.zeros((h, w), dtype=np.uint16)

    # analyze the binary patterns from the camera
    for i in range(0, 15):
        # read the file
        patt_gray = cv2.resize(cv2.imread("images/pattern%03d.jpg" % (i + 2), cv2.IMREAD_GRAYSCALE) / 255.0, (0, 0), fx=scale_factor, fy=scale_factor)

        # mask where the pixels are ON
        on_mask = (patt_gray > ref_on) & proj_mask
        on_mask_numbers = (on_mask*1).astype(np.uint16)

        # this code corresponds with the binary pattern code
        bit_code = np.uint16(1 << i)

        # TODO: populate scan_bits by putting the bit_code according to on_mask
        on_mask_numbers_scaled = np.multiply(on_mask_numbers, np.power(2,i)) # left shift
        scan_bits = scan_bits + on_mask_numbers_scaled  # add last 1 or 0

    print("load codebook")
    # the codebook translates from <binary code> to (x,y) in projector screen space
    with open("binary_codes_ids_codebook.pckl", "r") as f:
        binary_codes_ids_codebook = pickle.load(f)

    camera_points = []
    projector_points = []
    for x in range(w):
        for y in range(h):
            if not proj_mask[y, x]:
                continue  # no projection here
            if scan_bits[y, x] not in binary_codes_ids_codebook:
                continue  # bad binary code

            # TODO: use binary_codes_ids_codebook[...] and scan_bits[y,x] to
            # TODO: find for the camera (x,y) the projector (p_x, p_y).
            # TODO: store your points in camera_points and projector_points
            # (y,x) is a pixel in image ---> projector pixel

            projector_cr = binary_codes_ids_codebook[scan_bits[y, x]]
            if projector_cr[0] >= 1279 or projector_cr[1] >= 799:  # filter
                continue

            camera_points.append((x/2.0, y/2.0))  # 239455
            projector_points.append(binary_codes_ids_codebook[scan_bits[y, x]])  # 239455 241133

            # IMPORTANT!!! : due to differences in calibration and acquisition - divide the camera points by 2

    # now that we have 2D-2D correspondances, we can triangulate 3D points!
    projector_points_array = np.asarray(projector_points) # 239455 241133

    projector_points_x = projector_points_array[:,0].astype(np.float32)
    projector_points_y = projector_points_array[:,1].astype(np.float32)

    projector_points_x_norm = (projector_points_x.astype(np.float32))/1280
    projector_points_y_norm = (projector_points_y.astype(np.float32))/800

    cor_image = np.zeros((ref_white.shape[0],ref_white.shape[1], 3))

    for i in range(len(camera_points)):
        cor_image[int(camera_points[i][1]*2)][int(camera_points[i][0]*2)][0] = projector_points_x_norm[i] # RED
        cor_image[int(camera_points[i][1]*2)][int(camera_points[i][0]*2)][1] = projector_points_y_norm[i] # Green

    plt.imshow(cor_image)
    output_name_figure = sys.argv[1] + "correspondence.jpg"
    plt.savefig(output_name_figure)

    # load the prepared stereo calibration between projector and camera
    with open("stereo_calibration.pckl", "r") as f:
        d = pickle.load(f)
        camera_K = d['camera_K']
        camera_d = d['camera_d']
        projector_K = d['projector_K']
        projector_d = d['projector_d']
        projector_R = d['projector_R']
        projector_t = d['projector_t']

    # TODO: use cv2.undistortPoints to get normalized points for camera, use camera_K and camera_d
    camera_points_array = np.array([get_points_in_array_from_tuples(camera_points)])
    camera_points_fixed = cv2.undistortPoints(camera_points_array, camera_K, camera_d)

    # TODO: use cv2.undistortPoints to get normalized points for projector, use projector_K and projector_d
    projector_points_array = np.array([get_points_in_array_from_tuples(projector_points)])
    projector_points_fixed = cv2.undistortPoints(projector_points_array, projector_K, projector_d)

    # TODO: use cv2.triangulatePoints to triangulate the normalized points
    I3 = np.identity(3)
    zero_mat = np.zeros((3,1))
    P1 = np.hstack((I3, zero_mat))
    P2 = np.hstack((projector_R, projector_t))
    points_output = cv2.triangulatePoints(P1, P2, camera_points_fixed, projector_points_fixed)

    # TODO: use cv2.convertPointsFromHomogeneous to get real 3D points
    # TODO: name the resulted 3D points as "points_3d"
    points_3d = cv2.convertPointsFromHomogeneous(points_output.T) # 239455

    # apply another filter on the Z-component
    mask = (points_3d[:, :, 2] > 200) & (points_3d[:, :, 2] < 1400) # 78742

    j = 0
    points_3d_new = np.zeros((np.sum(mask),1, 3), np.float32)
    for i in range(points_3d.shape[0]):
        if(mask[i] == True):
            points_3d_new[j][0][0] = points_3d[i][0][0]
            points_3d_new[j][0][1] = points_3d[i][0][1]
            points_3d_new[j][0][2] = points_3d[i][0][2]
            j = j + 1

    return points_3d_new # 226427

def get_points_in_array_from_tuples(points_tuples):
    points_array = np.zeros((len(points_tuples), 2), dtype=np.float32)
    index = 0
    for point in points_tuples:
        x = point[0]
        y = point[1]
        points_array[index][0] = x
        points_array[index][1] = y
        index = index + 1
    return points_array

def write_3d_points(points_3d):
    # ===== DO NOT CHANGE THIS FUNCTION =====

    print("write output point cloud")
    print(points_3d.shape)
    output_name = sys.argv[1] + "output.xyz"
    with open(output_name, "w") as f:
        for p in points_3d:
            f.write("%d %d %d\n" % (p[0, 0], p[0, 1], p[0, 2]))

    return points_3d

def reconstruct_from_binary_patterns_bonus():
    scale_factor = 1.0
    ref_white = cv2.resize(cv2.imread("images/pattern000.jpg", cv2.IMREAD_GRAYSCALE) / 255.0, (0, 0),
                           fx=scale_factor,
                           fy=scale_factor)
    ref_black = cv2.resize(cv2.imread("images/pattern001.jpg", cv2.IMREAD_GRAYSCALE) / 255.0, (0, 0),
                           fx=scale_factor,
                           fy=scale_factor)
    ref_avg = (ref_white + ref_black) / 2.0

    ref_on = ref_avg + 0.05  # a threshold for ON pixels
    ref_off = ref_avg - 0.05  # add a small buffer region

    h, w = ref_white.shape

    # mask of pixels where there is projection
    proj_mask = (ref_white > (ref_black + 0.05))

    scan_bits = np.zeros((h, w), dtype=np.uint16)

    # analyze the binary patterns from the camera
    for i in range(0, 15):
        # read the file
        patt_gray = cv2.resize(cv2.imread("images/pattern%03d.jpg" % (i + 2), cv2.IMREAD_GRAYSCALE) / 255.0,
                               (0, 0), fx=scale_factor, fy=scale_factor)

        # mask where the pixels are ON
        on_mask = (patt_gray > ref_on) & proj_mask
        on_mask_numbers = (on_mask * 1).astype(np.uint16)

        # this code corresponds with the binary pattern code
        bit_code = np.uint16(1 << i)

        # TODO: populate scan_bits by putting the bit_code according to on_mask
        on_mask_numbers_scaled = np.multiply(on_mask_numbers, np.power(2, i))  # left shift
        scan_bits = scan_bits + on_mask_numbers_scaled  # add last 1 or 0

    print("load codebook")
    # the codebook translates from <binary code> to (x,y) in projector screen space
    with open("binary_codes_ids_codebook.pckl", "r") as f:
        binary_codes_ids_codebook = pickle.load(f)

    camera_points = []
    projector_points = []
    for x in range(w):
        for y in range(h):
            if not proj_mask[y, x]:
                continue  # no projection here
            if scan_bits[y, x] not in binary_codes_ids_codebook:
                continue  # bad binary code

            # TODO: use binary_codes_ids_codebook[...] and scan_bits[y,x] to
            # TODO: find for the camera (x,y) the projector (p_x, p_y).
            # TODO: store your points in camera_points and projector_points
            # (y,x) is a pixel in image ---> projector pixel

            projector_cr = binary_codes_ids_codebook[scan_bits[y, x]]
            if projector_cr[0] >= 1279 or projector_cr[1] >= 799:  # filter
                continue

            camera_points.append((x / 2.0, y / 2.0))  # 239455
            projector_points.append(binary_codes_ids_codebook[scan_bits[y, x]])  # 239455 241133

            # IMPORTANT!!! : due to differences in calibration and acquisition - divide the camera points by 2

    # now that we have 2D-2D correspondances, we can triangulate 3D points!
    projector_points_array = np.asarray(projector_points)  # 239455

    projector_points_x = projector_points_array[:, 0].astype(np.float32)
    projector_points_y = projector_points_array[:, 1].astype(np.float32)

    projector_points_x_norm = (projector_points_x.astype(np.float32)) / 1280
    projector_points_y_norm = (projector_points_y.astype(np.float32)) / 800

    cor_image = np.zeros((ref_white.shape[0], ref_white.shape[1], 3))

    for i in range(len(camera_points)):
        cor_image[int(camera_points[i][1] * 2)][int(camera_points[i][0] * 2)][0] = projector_points_x_norm[i]  # RED
        cor_image[int(camera_points[i][1] * 2)][int(camera_points[i][0] * 2)][1] = projector_points_y_norm[i]  # Green

    # load the prepared stereo calibration between projector and camera
    with open("stereo_calibration.pckl", "r") as f:
        d = pickle.load(f)
        camera_K = d['camera_K']
        camera_d = d['camera_d']
        projector_K = d['projector_K']
        projector_d = d['projector_d']
        projector_R = d['projector_R']
        projector_t = d['projector_t']

    # TODO: use cv2.undistortPoints to get normalized points for camera, use camera_K and camera_d
    camera_points_array = np.array([get_points_in_array_from_tuples(camera_points)])
    camera_points_fixed = cv2.undistortPoints(camera_points_array, camera_K, camera_d)

    # TODO: use cv2.undistortPoints to get normalized points for projector, use projector_K and projector_d
    projector_points_array = np.array([get_points_in_array_from_tuples(projector_points)])
    projector_points_fixed = cv2.undistortPoints(projector_points_array, projector_K, projector_d)

    # TODO: use cv2.triangulatePoints to triangulate the normalized points
    I3 = np.identity(3)
    zero_mat = np.zeros((3, 1))
    P1 = np.hstack((I3, zero_mat))
    P2 = np.hstack((projector_R, projector_t))
    points_output = cv2.triangulatePoints(P1, P2, camera_points_fixed, projector_points_fixed)

    # TODO: use cv2.convertPointsFromHomogeneous to get real 3D points
    # TODO: name the resulted 3D points as "points_3d"
    points_3d = cv2.convertPointsFromHomogeneous(points_output.T)  # 239455

    ref_image = cv2.imread("images/pattern001.jpg", cv2.IMREAD_COLOR)
    # create map of colors
    points_color = []
    for i in range(len(camera_points)):
        camera_X = int(camera_points[i][0]*2)
        camera_Y = int(camera_points[i][1]*2)
        points_color.append((ref_image[camera_Y][camera_X][2], ref_image[camera_Y][camera_X][1], ref_image[camera_Y][camera_X][0]))

    # apply another filter on the Z-component
    mask = (points_3d[:, :, 2] > 200) & (points_3d[:, :, 2] < 1400)  # 78742

    j = 0
    points_3d_new = np.zeros((np.sum(mask), 1, 3), np.float32)
    points_color_new = []
    for i in range(points_3d.shape[0]):
        if (mask[i] == True):
            points_3d_new[j][0][0] = points_3d[i][0][0]
            points_3d_new[j][0][1] = points_3d[i][0][1]
            points_3d_new[j][0][2] = points_3d[i][0][2]
            points_color_new.append(points_color[i])
            j = j + 1

    return points_3d_new, points_color_new   # 226427

def write_3d_points_bonus(points_3d_new, points_color_new):
    # ===== DO NOT CHANGE THIS FUNCTION =====

    print("write output point cloud for bonus")
    print(points_3d_new.shape)
    output_name = sys.argv[1] + "output_color.xyz"
    index = 0
    with open(output_name, "w") as f:
        for p in points_3d_new:
            f.write("%d %d %d %d %d %d\n" % (p[0, 0], p[0, 1], p[0, 2], points_color_new[index][0], points_color_new[index][1], points_color_new[index][2]))
            index = index + 1

    return points_3d_new, points_color_new


if __name__ == '__main__':

    # ===== DO NOT CHANGE THIS FUNCTION =====

    # validate the input arguments
    if (len(sys.argv) != 2):
        help_message()
        sys.exit()

    points_3d = reconstruct_from_binary_patterns()
    write_3d_points(points_3d)
    points_3d_new, points_color_new = reconstruct_from_binary_patterns_bonus()
    write_3d_points_bonus(points_3d_new, points_color_new)