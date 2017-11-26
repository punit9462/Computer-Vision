# ================================================
# Skeleton codes for HW4
# Read the skeleton codes carefully and put all your
# codes into main function
# ================================================

import sys
import cv2
import numpy as np
from skimage.segmentation import slic
import maxflow
from scipy.spatial import Delaunay


def help_message():
    print("Usage: [Input_Image]")
    print("[Input_Image]")
    print("Path to the input image")
    print("Example usages:")
    print(sys.argv[0] + " astronaut.png")


# Calculate the SLIC superpixels, their histograms and neighbors
def superpixels_histograms_neighbors(img):
    # SLIC
    segments = slic(img, n_segments=500, compactness=24)
    segments_ids = np.unique(segments)

    # centers
    centers = np.array(
        [np.mean(np.nonzero(segments == i), axis=1) for i in segments_ids])

    # H-S histograms for all superpixels
    hsv = cv2.cvtColor(img.astype('float32'), cv2.COLOR_BGR2HSV)
    bins = [20, 20]  # H = S = 20
    ranges = [0, 360, 0, 1]  # H: [0, 360], S: [0, 1]
    colors_hists = np.float32([
        cv2.calcHist([hsv], [0, 1], np.uint8(segments == i), bins,
                     ranges).flatten() for i in segments_ids
    ])

    # neighbors via Delaunay tesselation
    tri = Delaunay(centers)

    return (centers, colors_hists, segments, tri.vertex_neighbor_vertices)


# Get superpixels IDs for FG and BG from marking
def find_superpixels_under_marking(superpixels):
    fg_segments = np.unique(superpixels[fg_pixels])
    bg_segments = np.unique(superpixels[bg_pixels])
    return (fg_segments, bg_segments)


# Sum up the histograms for a given selection of superpixel IDs, normalize
def cumulative_histogram_for_superpixels(ids, histograms):
    h = np.sum(histograms[ids], axis=0)
    return h / h.sum()


# Get a bool mask of the pixels for a given selection of superpixel IDs
def pixels_for_segment_selection(superpixels_labels, selection):
    pixels_mask = np.where(np.isin(superpixels_labels, selection), True, False)
    return pixels_mask


# Get a normalized version of the given histograms (divide by sum)
def normalize_histograms(histograms):
    return np.float32([h / h.sum() for h in histograms])


# Perform graph cut using superpixels histograms
def do_graph_cut(fgbg_hists, fgbg_superpixels, norm_hists, neighbors):
    num_nodes = norm_hists.shape[0]
    # Create a graph of N nodes, and estimate of 5 edges per node
    g = maxflow.Graph[float](num_nodes, num_nodes * 5)
    # Add N nodes
    nodes = g.add_nodes(num_nodes)

    hist_comp_alg = cv2.HISTCMP_KL_DIV

    # Smoothness term: cost between neighbors
    indptr, indices = neighbors
    for i in range(len(indptr) - 1):
        N = indices[indptr[i]:indptr[i + 1]]  # list of neighbor superpixels
        hi = norm_hists[i]  # histogram for center
        for n in N:
            if (n < 0) or (n > num_nodes):
                continue
            # Create two edges (forwards and backwards) with capacities based on
            # histogram matching
            hn = norm_hists[n]  # histogram for neighbor
            g.add_edge(nodes[i], nodes[n],
                       20 - cv2.compareHist(hi, hn, hist_comp_alg),
                       20 - cv2.compareHist(hn, hi, hist_comp_alg))

    # Match term: cost to FG/BG
    for i, h in enumerate(norm_hists):
        if i in fgbg_superpixels[0]:
            g.add_tedge(nodes[i], 0, 1000)  # FG - set high cost to BG
        elif i in fgbg_superpixels[1]:
            g.add_tedge(nodes[i], 1000, 0)  # BG - set high cost to FG
        else:
            g.add_tedge(nodes[i],
                        cv2.compareHist(fgbg_hists[0], h, hist_comp_alg),
                        cv2.compareHist(fgbg_hists[1], h, hist_comp_alg))

    g.maxflow()
    return g.get_grid_segments(nodes)


def RMSD(target, master):
    # Note: use grayscale images only

    # Get width, height, and number of channels of the master image
    master_height, master_width = master.shape[:2]
    master_channel = len(master.shape)

    # Get width, height, and number of channels of the target image
    target_height, target_width = target.shape[:2]
    target_channel = len(target.shape)

    # Validate the height, width and channels of the input image
    if (master_height != target_height or master_width != target_width or
            master_channel != target_channel):
        return -1
    else:

        total_diff = 0.0
        dst = cv2.absdiff(master, target)
        dst = cv2.pow(dst, 2)
        mean = cv2.mean(dst)
        total_diff = mean[0]**(1 / 2.0)

        return total_diff


drawing = False  # true if mouse is pressed
mode = True  # if True, background otherwise foreground
ix, iy = -1, -1
num_of_lines = 0
fg_pixels = []
bg_pixels = []


# mouse callback function
def draw_circle(event, x, y, flags, param):

    global ix, iy, drawing, mode, num_of_lines, fg_pixels, bg_pixels

    if event == cv2.EVENT_LBUTTONDOWN:  # background
        drawing = True
        mode = True
        bg_pixels[y][x] = True
        bg_pixels[y + 1][x + 1] = True
        bg_pixels[y + 2][x + 2] = True
        bg_pixels[y + 3][x + 3] = True
        bg_pixels[y + 4][x + 4] = True
        bg_pixels[y + 5][x + 5] = True
        bg_pixels[y - 1][x - 1] = True
        bg_pixels[y - 2][x - 2] = True
        bg_pixels[y - 3][x - 3] = True
        bg_pixels[y - 4][x - 4] = True
        bg_pixels[y - 5][x - 5] = True
        ix, iy = x, y

    elif event == cv2.EVENT_RBUTTONDOWN:  # foreground
        drawing = True
        mode = False
        fg_pixels[y][x] = True
        fg_pixels[y + 1][x + 1] = True
        fg_pixels[y + 2][x + 2] = True
        fg_pixels[y + 3][x + 3] = True
        fg_pixels[y + 4][x + 4] = True
        fg_pixels[y + 5][x + 5] = True
        fg_pixels[y - 1][x - 1] = True
        fg_pixels[y - 2][x - 2] = True
        fg_pixels[y - 3][x - 3] = True
        fg_pixels[y - 4][x - 4] = True
        fg_pixels[y - 5][x - 5] = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                cv2.line(img, (ix, iy), (x, y), (0, 255, 0), 8)  # background
                bg_pixels[y][x] = True
                bg_pixels[y + 1][x + 1] = True
                bg_pixels[y + 2][x + 2] = True
                bg_pixels[y + 3][x + 3] = True
                bg_pixels[y + 4][x + 4] = True
                bg_pixels[y + 5][x + 5] = True
                bg_pixels[y - 1][x - 1] = True
                bg_pixels[y - 2][x - 2] = True
                bg_pixels[y - 3][x - 3] = True
                bg_pixels[y - 4][x - 4] = True
                bg_pixels[y - 5][x - 5] = True
            else:
                cv2.line(img, (ix, iy), (x, y), (0, 0, 255), 8)  # foreground
                fg_pixels[y][x] = True
                fg_pixels[y + 1][x + 1] = True
                fg_pixels[y + 2][x + 2] = True
                fg_pixels[y + 3][x + 3] = True
                fg_pixels[y + 4][x + 4] = True
                fg_pixels[y + 5][x + 5] = True
                fg_pixels[y - 1][x - 1] = True
                fg_pixels[y - 2][x - 2] = True
                fg_pixels[y - 3][x - 3] = True
                fg_pixels[y - 4][x - 4] = True
                fg_pixels[y - 5][x - 5] = True
            ix = x
            iy = y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(img, (ix, iy), (x, y), (0, 255, 0), 8)
        bg_pixels[y][x] = True
        bg_pixels[y + 1][x + 1] = True
        bg_pixels[y + 2][x + 2] = True
        bg_pixels[y + 3][x + 3] = True
        bg_pixels[y + 4][x + 4] = True
        bg_pixels[y + 5][x + 5] = True
        bg_pixels[y - 1][x - 1] = True
        bg_pixels[y - 2][x - 2] = True
        bg_pixels[y - 3][x - 3] = True
        bg_pixels[y - 4][x - 4] = True
        bg_pixels[y - 5][x - 5] = True
        ix = x
        iy = y
        num_of_lines = num_of_lines + 1
        calculate(param[0], param[1], param[2], param[3])

    elif event == cv2.EVENT_RBUTTONUP:
        drawing = False
        cv2.line(img, (ix, iy), (x, y), (0, 0, 255), 8)
        fg_pixels[y][x] = True
        fg_pixels[y + 1][x + 1] = True
        fg_pixels[y + 2][x + 2] = True
        fg_pixels[y + 3][x + 3] = True
        fg_pixels[y + 4][x + 4] = True
        fg_pixels[y + 5][x + 5] = True
        fg_pixels[y - 1][x - 1] = True
        fg_pixels[y - 2][x - 2] = True
        fg_pixels[y - 3][x - 3] = True
        fg_pixels[y - 4][x - 4] = True
        fg_pixels[y - 5][x - 5] = True
        ix = x
        iy = y
        num_of_lines = num_of_lines + 1
        calculate(param[0], param[1], param[2], param[3])


def calculate(color_hists, superpixels, neighbors, norm_hists):

    fg_segments, bg_segments = find_superpixels_under_marking(superpixels)
    fg_cumulative_hist = cumulative_histogram_for_superpixels(
        fg_segments, color_hists)
    bg_cumulative_hist = cumulative_histogram_for_superpixels(
        bg_segments, color_hists)
    fgbg_hists = np.vstack((fg_cumulative_hist, bg_cumulative_hist))

    diff = abs(fg_segments.shape[0] - bg_segments.shape[0])
    if diff != 0:
        dummy = np.full(diff, -1, dtype=np.int64)  # diff array
        if fg_segments.shape[0] < bg_segments.shape[0]:
            fg_segments = np.append(fg_segments, dummy)
        else:
            bg_segments = np.append(bg_segments, dummy)

    fgbg_superpixels = np.vstack((fg_segments, bg_segments))

    graph_cut = do_graph_cut(fgbg_hists, fgbg_superpixels, norm_hists,
                             neighbors)

    segmask = pixels_for_segment_selection(superpixels, np.nonzero(graph_cut))
    segmask = np.uint8(segmask * 255)
    if num_of_lines >= 1:
        cv2.imshow('Segmentation', segmask)


if __name__ == '__main__':

    np.seterr(divide='ignore', invalid='ignore')

    # validate the input arguments
    if (len(sys.argv) != 2):
        help_message()
        sys.exit()

    # ======================================== #
    # write all your codes here
    img = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
    centers, color_hists, superpixels, neighbors = superpixels_histograms_neighbors(
        img)
    norm_hists = normalize_histograms(color_hists)
    fg_pixels = np.full(
        (superpixels.shape[0], superpixels.shape[1]), False, dtype=bool)
    bg_pixels = np.full(
        (superpixels.shape[0], superpixels.shape[1]), False, dtype=bool)

    # Create a window and bind the function to window
    cv2.namedWindow('image')
    cv2.setMouseCallback(
        'image',
        draw_circle,
        param=[color_hists, superpixels, neighbors, norm_hists])

    while (1):
        cv2.imshow('image', img)
        if cv2.waitKey(20) & 0xFF == 27:
            break
    cv2.destroyAllWindows()

    # ======================================== #

    # read video file
    # output_name = sys.argv[3] + "mask.png"
    # cv2.imwrite(output_name, mask);
