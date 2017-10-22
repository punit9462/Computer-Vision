# Instructions:
# For question 1, only modify function: histogram_equalization
# For question 2, only modify functions: low_pass_filter, high_pass_filter, deconvolution
# For question 3, only modify function: laplacian_pyramid_blending

import os
import sys
import cv2
import numpy as np

def help_message():
   print("Usage: [Question_Number] [Input_Options] [Output_Options]")
   print("[Question Number]")
   print("1 Histogram equalization")
   print("2 Frequency domain filtering")
   print("3 Laplacian pyramid blending")
   print("[Input_Options]")
   print("Path to the input images")
   print("[Output_Options]")
   print("Output directory")
   print("Example usages:")
   print(sys.argv[0] + " 1 " + "[path to input image] " + "[output directory]")                                # Single input, single output
   print(sys.argv[0] + " 2 " + "[path to input image1] " + "[path to input image2] " + "[output directory]")   # Two inputs, three outputs
   print(sys.argv[0] + " 3 " + "[path to input image1] " + "[path to input image2] " + "[output directory]")   # Two inputs, single output
   
# ===================================================
# ======== Question 1: Histogram equalization =======
# ===================================================


def equalize(one_chan_image_b):
    hist, bins = np.histogram(one_chan_image_b, bins=256)
    cdf = hist.cumsum()

    # Normalization
    cdf_new = cdf.copy()
    cdf_m = np.ma.masked_equal(cdf_new, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf_new = np.ma.filled(cdf_m, 0).astype('uint8')
    return cdf_new[one_chan_image_b]


def histogram_equalization(img_in):
   # Write histogram equalization here

   one_chan_image_b, one_chan_image_g, one_chan_image_r = cv2.split(img_in)  # split in channels

   equal_one_chan_image_b = equalize(one_chan_image_b)
   equal_one_chan_image_g = equalize(one_chan_image_g)
   equal_one_chan_image_r = equalize(one_chan_image_r)

   img_out = cv2.merge((equal_one_chan_image_b, equal_one_chan_image_g, equal_one_chan_image_r))
   return True, img_out


def Question1():

   # Read in input images
   input_image = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR);

   # Histogram equalization
   succeed, output_image = histogram_equalization(input_image)
   
   # Write out the result
   output_name = sys.argv[3] + "1.jpg"
   cv2.imwrite(output_name, output_image)

   # Uncomment the following to compare the histograms
   # input_image = cv2.imread(sys.argv[2], 0);
   # output_image = cv2.equalizeHist(input_image)
   # cv2.imwrite("test.jpg", output_image)

   return True
   
# ===================================================
# ===== Question 2: Frequency domain filtering ======
# ===================================================


def low_pass_filter(img_in):
	
   # Write low pass filter here

   dft = np.fft.fft2(img_in)
   dft_shift = np.fft.fftshift(dft)

   rows, cols = img_in.shape
   crow, ccol = rows / 2, cols / 2

   # create a mask first, center square is 1, remaining all zeros
   mask = np.zeros((rows, cols), np.uint8)
   mask[crow - 20:crow + 20, ccol - 20:ccol + 20] = 1

   # apply mask and inverse DFT
   fshift = dft_shift * mask
   f_ishift = np.fft.ifftshift(fshift)
   img_back = np.fft.ifft2(f_ishift)
   img_out = np.uint8(np.abs(img_back))
   return True, img_out


def high_pass_filter(img_in):

   # Write high pass filter here
   f = np.fft.fft2(img_in)
   fshift = np.fft.fftshift(f)

   rows, cols = img_in.shape
   crow, ccol = rows / 2, cols / 2
   fshift[crow - 20:crow + 20, ccol - 20:ccol + 20] = 0

   f_ishift = np.fft.ifftshift(fshift)
   img_back = np.fft.ifft2(f_ishift)
   img_out = np.uint8(np.abs(img_back))

   return True, img_out


def ft(im, newsize=None):
    dft = np.fft.fft2(np.float32(im),newsize)
    return np.fft.fftshift(dft)


def ift(shift):
    f_ishift = np.fft.ifftshift(shift)
    img_back = np.fft.ifft2(f_ishift)
    return np.abs(img_back)


def deconvolution(img_in):
   
   # Write deconvolution codes here

   gk = cv2.getGaussianKernel(21, 5)
   gk = gk * gk.T

   img_f = ft(img_in, (img_in.shape[0], img_in.shape[1]))
   gk_f = ft(gk, (img_in.shape[0], img_in.shape[1]))

   result_image_f = img_f / gk_f
   img_out = ift(result_image_f)
   img_out = np.uint8(img_out * 255)
   return True, img_out


def Question2():

   # Read in input images
   input_image1 = cv2.imread(sys.argv[2], 0)  # Used for HPF and LPF
   input_image2 = cv2.imread(sys.argv[3], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)  # Used for deconvolution

   # Low and high pass filter
   succeed1, output_image1 = low_pass_filter(input_image1)
   succeed2, output_image2 = high_pass_filter(input_image1)
   
   # Deconvolution
   succeed3, output_image3 = deconvolution(input_image2)
   
   # Write out the result
   output_name1 = sys.argv[4] + "2.jpg"
   output_name2 = sys.argv[4] + "3.jpg"
   output_name3 = sys.argv[4] + "4.jpg"
   cv2.imwrite(output_name1, output_image1)
   cv2.imwrite(output_name2, output_image2)
   cv2.imwrite(output_name3, output_image3)
   
   return True
   
# ===================================================
# ===== Question 3: Laplacian pyramid blending ======
# ===================================================


def get_gaussian_pyramid(x, level):
    G = x.copy()
    gp = [G]
    for i in xrange(level):
        G = cv2.pyrDown(G)
        gp.append(G)
    return gp


def get_laplacian_pyramid(gp, level):
    lp = [gp[level-1]]
    for i in xrange(level-1, 0, -1):
        GE = cv2.pyrUp(gp[i])
        L = cv2.subtract(gp[i - 1], GE)
        lp.append(L)
    return lp


def laplacian_pyramid_blending(img_in1, img_in2):

   # Write laplacian pyramid blending codes here

   gpA = get_gaussian_pyramid(img_in1, 6)
   gpB = get_gaussian_pyramid(img_in2, 6)
   lpA = get_laplacian_pyramid(gpA, 6)
   lpB = get_laplacian_pyramid(gpB, 6)

   # Now add left and right halves of images in each level
   LS = []
   for la, lb in zip(lpA, lpB):
       rows, cols, dpt = la.shape
       ls = np.hstack((la[:, 0:cols / 2], lb[:, cols / 2:]))
       LS.append(ls)

   # now reconstruct
   img_out = LS[0]
   for i in xrange(1, 6):
       img_out = cv2.pyrUp(img_out)
       img_out = cv2.add(img_out, LS[i])

   return True, img_out

def Question3():

   # Read in input images
   input_image1 = cv2.imread(sys.argv[2], cv2.COLOR_BGR2RGB);
   input_image2 = cv2.imread(sys.argv[3], cv2.COLOR_BGR2RGB);

   input_image1 = input_image1[:, :input_image1.shape[0]]
   input_image2 = input_image2[:input_image1.shape[0], :input_image1.shape[0]]

   # Laplacian pyramid blending
   succeed, output_image = laplacian_pyramid_blending(input_image1, input_image2)
   
   # Write out the result
   output_name = sys.argv[4] + "5.jpg"
   cv2.imwrite(output_name, output_image)
   
   return True


if __name__ == '__main__':

    question_number = -1

    # Validate the input arguments
    if (len(sys.argv) < 4):
        help_message()
        sys.exit()
    else:
        question_number = int(sys.argv[1])

        if (question_number == 1 and not (len(sys.argv) == 4)):
            help_message()
            sys.exit()
        if (question_number == 2 and not (len(sys.argv) == 5)):
            help_message()
            sys.exit()
        if (question_number == 3 and not (len(sys.argv) == 5)):
            help_message()
            sys.exit()
        if (question_number > 3 or question_number < 1 or len(sys.argv) > 5):
            print("Input parameters out of bound ...")
            sys.exit()

    function_launch = {
        1: Question1,
        2: Question2,
        3: Question3,
    }

    # Call the function
    function_launch[question_number]()
