# Coded by Bolin Qin, Center for MRI Research, Peking University, Beijing, China
# Ref: 
# Report on a Multicenter fMRI Quality Assurance Protocol
# Dartmouth fMRI QA: https://www.dartmouth.edu/dbic/research_infrastructure/qualityassurance.html
# Edited on 2023.10.21

# Useful packages========================================================================================================
from tkinter import Tk, filedialog   # input the QA.nii.gz data 
from scipy.ndimage import uniform_filter  # 用于均值滤波
from matplotlib.patches import Rectangle  # Draw ROI area
import numpy as np
import nibabel as nib
import re
from skimage.filters import threshold_otsu
from skimage.morphology import binary_dilation
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression   # detrend
import math
from scipy.stats import norm  # for normal distribution regression of the time course
import cv2
from reportlab.pdfgen import canvas
from PyPDF2 import PdfMerger
import os
import time


plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 20
plt.style.use(['seaborn-bright', 'seaborn-paper'])

#============================================================================================================================
# Generate a GUI to choose QA.nii.gz
def select_data_file():
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    data_file = filedialog.askopenfilename(title="Select NIfTI data file for analysis",
                                           filetypes=(("NIfTI files", "*.nii.gz;*.nii"),("All files", "*")))
    root.destroy()
    if not data_file:
        raise ValueError("Error: input file not given.")
    return data_file


# Generate a GUI to choose output folder
def select_data_directory():
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    data_directory = filedialog.askdirectory(title="Select a directory for output results")
    root.destroy()
    if not data_directory:
        raise ValueError("Error: directory not selected.")
    return data_directory


# Get the slice format of the pixel range
def get_pixel_range(center, width):
    ROI_half_start = math.floor(width / 2)
    ROI_half_end = math.ceil(width / 2)
    start = math.floor(center - ROI_half_start)
    end = math.floor(center + ROI_half_end)
    return slice(start, end)

# Detrend the time-series y using matrix X, output the residuals
def detrend_fast(y, X):
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    residual = y - y_pred
    return residual


def cal_CV(x_range,y_range):
    # Get ROI: data_raw(x,y,t) --> slice_data_ROI(x,y,t)
    slice_data_ROI = data_raw[x_range, y_range, :]

    # Time-series of mean signal intensity(t)
    mean_sig_intensity_t = np.mean(slice_data_ROI, axis=(0,1))

    # Detrend to get the residuals: mean_sig_intensity_t_dt(t)
    poly_det_ord = 3
    X = np.vander(np.arange(1, dyns+1), poly_det_ord+1, increasing=True)
    mean_sig_intensity_t_dt = np.apply_along_axis(detrend_fast, 0, mean_sig_intensity_t, X)

    # CV = (SD of a time-series)/(mean signal intensity)
    CV_value = np.std(mean_sig_intensity_t_dt) / np.mean(mean_sig_intensity_t)

    return CV_value

# 0. Title of the Report ====================================================================================================================================
title = "QA fMRI Report"
date = '20231020'
coil_type = 'Nova 8Tx/32Rx Head'


# 1. Import the QA data using GUI ============================================================================================================================
data_file = select_data_file()
basename = re.sub(r"\.nii\.gz$", "", data_file)
basename = re.sub(r"\.nii$", "", basename)
basename = basename.split('/')
basename = basename[-1]
print('basename = ',basename)

# Output file direction
output_setting = select_data_directory()
output_dir = output_setting + '/' + basename
print('Output folder directory: ',output_setting)
data_nii = nib.load(data_file)
print("Reading data  : ", data_file)

# nd.array-form: data(x,y,z,t)
data = data_nii.get_fdata()
data = np.transpose(data,(1,0,2,3))  # data(x,y,z,t)
x_dim,y_dim,z_dim= data.shape[0],data.shape[1],data.shape[2]
N = data.shape[3]  # total number of volumes, 200

# Read resolutions from header of data_nii
header = data_nii.header
resolution = header.get_zooms()
x_pix_dim, y_pix_dim, z_pix_dim = resolution[0],resolution[1],resolution[2]
tr = resolution[3]

# Choose the middle slice, abandon the first skip=2 volumes: data (x,y,z,t) --> data_raw(x,y,t)
slice_num = int(np.ceil(data.shape[2] / 2)-1)
skip = 2  
dyns = N - skip
data_raw = data[:, :, slice_num, skip:N]

# Set ROI width
roi_width = 21

# Print the basic analysis parameters
print("Basic analysis parameters")
print("-------------------------")
print("X,Y matrix       : {} x {}".format(x_dim, y_dim))
print("Slices           : {}".format(z_dim))
print("(X,Y,Z) pix dims : ({:.2f}, {:.2f}, {:.2f}) mm".format(x_pix_dim, y_pix_dim, z_pix_dim))
print("TR               : {:.2f} s".format(tr))
print("Ref Slice Index  : {}".format(slice_num))
print("ROI width        : {}".format(roi_width))
print("Total vols       : {}".format(N))
print("Analysis vols    : {}".format(dyns))

#2. ROI =======================================================================================================
# Average along time: data_raw(x,y,t) --> time_aver_image(x,y)
time_aver_image = np.nanmean(data_raw,axis=2)

# Threshold to grasp the phantom object: time_aver_image(x,y) --> mask_threshold(x,y) [bool]
mask_threshold = time_aver_image > threshold_otsu(time_aver_image)
# Close calculation kernel with size, in order to fill the holes in the mask
close_kernel_size = 5
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_kernel_size, close_kernel_size))  
mask_threshold = cv2.morphologyEx(mask_threshold.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

# Get the position of Center of Mass
x_pos = round(np.sum(([np.arange(x_dim)]*y_dim)*mask_threshold)/np.sum(mask_threshold))
y_pos = round(np.sum(np.transpose([np.arange(y_dim)] * x_dim) * mask_threshold) / np.sum(mask_threshold))

# Get the slice of x-dim and y-dim in the ROI
ROI_x = get_pixel_range(y_pos, roi_width)
ROI_y = get_pixel_range(x_pos, roi_width)
rect = Rectangle((ROI_y.start, ROI_x.start), roi_width, roi_width, linewidth=2, edgecolor='w', facecolor='none')

# Take the ROI area
time_aver_image_roi = time_aver_image[ROI_x, ROI_y]

# 3. Analysis ==========================================================================================================
# 3.1 Signal Map
time_aver_image = np.nanmean(data_raw,axis=2)
aver_signal = np.mean(time_aver_image[ROI_x, ROI_y])
print('Mean Signal Intensity = ', aver_signal)

# 3.2 Temporal Fluctuation Noise (TFN) Map
poly_det_ord = 3
X = np.vander(np.arange(1, dyns+1), poly_det_ord+1, increasing=True)
data_detrend = np.apply_along_axis(detrend_fast, 2, data_raw, X)
TFN_full = np.std(data_detrend, axis=2)
aver_TFN = np.mean(TFN_full[ROI_x, ROI_y])
print('TFN summary value = ',aver_TFN)

# 3.3 Signal-to-Fluctuation-Noise Ratio (SFNR) Map
SFNR_full = np.nan_to_num(time_aver_image / TFN_full)
aver_SFNR = np.mean(SFNR_full[ROI_x, ROI_y])
print('SFNR summary value = ',aver_SFNR)

# 3.4 Temporal Signal-to-Noise Ratio (tSNR) Map
time_sd_image = np.std(data_raw, axis=2)
tSNR_full = np.nan_to_num(time_aver_image / time_sd_image)
aver_tSNR = np.mean(tSNR_full[ROI_x, ROI_y])
print('tSNR summary value = ',aver_tSNR)

# 3.5 Static Spatial Noise (SSN, DIFF) Map
odd_dynamics = data_raw[:, :, ::2]
even_dynamics = data_raw[:, :, 1::2]
if odd_dynamics.shape[2] > even_dynamics.shape[2]:
    odd_dynamics = odd_dynamics[:, :, :-1]
    print("Odd number of dynamic scans, removing the last one for the odd-even diff calculation.")
DIFF = np.sum(odd_dynamics, axis=2) - np.sum(even_dynamics, axis=2)

# 3.6 Signal-to-Noise (SNR) Map
SNR_full = time_aver_image/np.std(DIFF)*np.sqrt(dyns)
variance_summary_value = np.std(DIFF[ROI_x, ROI_y]) ** 2
signal_summary_value = np.nanmean(time_aver_image[ROI_x, ROI_y])
SNR = signal_summary_value / np.sqrt(variance_summary_value / dyns)
print('SNR summary value = ',SNR)

# 3.7 Percent fluctuation and Drift
slice_data_ROI = data_raw[ROI_x, ROI_y, :]
mean_sig_intensity_t = np.mean(slice_data_ROI, axis=(0,1))
poly_det_ord = 3
X = np.vander(np.arange(1, dyns+1), poly_det_ord+1, increasing=True)
mean_sig_intensity_t_detrend = detrend_fast(mean_sig_intensity_t, X)
y_fit = mean_sig_intensity_t - mean_sig_intensity_t_detrend
sd_roi = np.std(mean_sig_intensity_t_detrend)
mean_sig_intensity = np.mean(mean_sig_intensity_t)
percent_fluc = 100.0 * sd_roi / mean_sig_intensity
print('Percent fluctuation = ',percent_fluc,'%')
# Drift of fit signal
percent_drift_fit = 100.0 * (np.max(y_fit) - np.min(y_fit)) / mean_sig_intensity
print('Drift of fit signal = ',percent_drift_fit,'%')
# Drift of raw signal
percent_drift = 100.0 * (np.max(mean_sig_intensity_t) - np.min(mean_sig_intensity_t)) / mean_sig_intensity
print('Drift of raw signal = ',percent_drift,'%')
norm_mu, norm_std = norm.fit(mean_sig_intensity_t_detrend)
norm_halfwidth = max(abs(np.floor(min(mean_sig_intensity_t_detrend))), abs(np.ceil(max(mean_sig_intensity_t_detrend))))
norm_x = np.linspace(-norm_halfwidth,norm_halfwidth, 100)
norm_y = norm.pdf(norm_x, norm_mu, norm_std)

# 3.8 Fourier Analysis
spec = np.abs(np.fft.fft(mean_sig_intensity_t_detrend))
freq = np.fft.fftfreq(len(spec), tr)
spec = spec[:len(spec)//2]
freq = freq[:len(freq)//2]
# MAD, Median Absolute Deviation
max_spec_outlier = np.max(spec) / np.median(np.abs(spec - np.median(spec)))
print('Median Absolute Deviation (MAD) = ',max_spec_outlier)
# Find the largest spec and its freq
peak_freq_index = np.argmax(spec)
peak_freq = freq[peak_freq_index]
print('Peak Freq = ',peak_freq,'Hz')

# 3.9 Weisskoff Analysis
# CV: coefficient of variance
CV = np.zeros(roi_width)
CV_ideal = np.zeros(roi_width)
for n in range(1, roi_width+1):
    x_range = get_pixel_range(y_pos, n)
    y_range = get_pixel_range(x_pos, n)
    CV[n-1] = cal_CV(x_range,y_range)
    CV_ideal[n-1] = CV[0]/n
# RDC: Radius of DeCorrelation
RDC = CV[0] / CV[roi_width-1]
print('Radius of DeCorrelation (RDC) = ',RDC)

# 3.10 Ghosting
pix_dim = min(x_pix_dim, y_pix_dim)
bg_shrink = 10
original_mask = binary_dilation(mask_threshold, np.ones((round(bg_shrink / pix_dim), round(bg_shrink / pix_dim))))
# Ghost mask
ghost_up=original_mask[:int(x_dim/2),:]
ghost_down=original_mask[int(x_dim/2):,:]
ghost_mask=np.vstack((ghost_down, ghost_up))
# Mean signal: original=1, ghost=0
signal_no_ghost_mask = original_mask.copy()
signal_no_ghost_mask[ghost_mask]=False
signal_no_ghost = time_aver_image*signal_no_ghost_mask
mean_signal = np.nanmean(signal_no_ghost)
# Mean ghost: original=0, ghost=1
ghost_no_signal_mask = ghost_mask.copy()
ghost_no_signal_mask[original_mask]=False
ghost_no_signal = time_aver_image*ghost_no_signal_mask
mean_ghost = np.nanmean(ghost_no_signal)
ghost_rate=100*mean_ghost/mean_signal
print('Ghost rate = ',ghost_rate,'%')
ghost_rate_t = []
for t in range(dyns):
    image_dyns = data_raw[:,:,t]
    signal_dyns = image_dyns*signal_no_ghost_mask
    mean_signal = np.nanmean(signal_dyns)
    ghost_dyns = image_dyns*ghost_no_signal_mask
    mean_ghost = np.nanmean(ghost_dyns)
    ghost_rate_t.append(100*mean_ghost/mean_signal)

# 3.11 Background Noise
background_mask = original_mask.copy()
background_mask[ghost_mask]=True
bg_shrink = 25
background_mask = binary_dilation(background_mask, np.ones((round(bg_shrink / pix_dim), round(bg_shrink / pix_dim))))
background_mask = np.logical_not(background_mask)
background = time_aver_image*background_mask
mean_background = np.nanmean(background)
mean_background_rate = mean_background/mean_signal*100
print('Mean background rate = ',mean_background_rate,'%')
background_t = []
for t in range(dyns):
    image_dyns = data_raw[:,:,t]
    background_dyns = image_dyns*background_mask
    mean_background = np.nanmean(background_dyns)
    background_t.append(mean_background)


# 4. Summary Report ============================================================================================================================
# 4.1 CSV: _results.csv
csv_file = output_dir + "_results.csv"
results_tab = pd.DataFrame({
        'Direction': [data_file],
        'Date': [date],
        'Coil': [coil_type],
        'Mean_Signal_Intensity': [mean_sig_intensity],
        'Peak_Freq_Hz': [peak_freq],
        'STD': [sd_roi],
        'SNR': [SNR],
        'TFN': [aver_TFN],
        'SFNR': [aver_SFNR],
        'tSNR': [aver_tSNR],
        'Percent_Fluctuation': [percent_fluc],
        'Drift': [percent_drift],
        'Drift_Fit': [percent_drift_fit],
        'RDC': [RDC],
        'Spectrum_MAD': [max_spec_outlier],
        'Mean_Ghost': [mean_ghost],
        'Ghost_Rate': [ghost_rate],
        'Mean_Background': [mean_background],
        'Background_Rate': [mean_background_rate]
    })
results_tab.to_csv(csv_file, index=False)

# 4.2 PDF: _report.pdf
# 4.2.1 Text
text_direction = output_dir + '_text.pdf'
pdf = canvas.Canvas(text_direction)
pdf.setFont("Courier", 10)
pdf_width, pdf_height = pdf._pagesize
title_width = pdf.stringWidth(title, "Courier", 10)
x,y = (pdf_width - title_width)/2, pdf_height - 50
pdf.drawString(x,y, title)
text_content=["",
              "\n Information:",
              "Date   : {}".format(date),
              "Coil   : {}".format(coil_type),
              "",
              "\n Basic analysis parameters:",
              "Matrix Size      : {} x {}".format(x_dim, y_dim),
              "Slices           : {}".format(z_dim),
              "Resolution       : {:.2f} x {:.2f} x {:.2f} mm3".format(x_pix_dim, y_pix_dim, z_pix_dim),
              "TR               : {} ms".format(int(tr*1000)),
              "Ref Slice Num    : {}".format(slice_num),
              "ROI width        : {}".format(roi_width),
              "Total vols       : {}".format(N),
              "Analysis vols    : {}".format(dyns),
              '',
              "\n QA metrics",
              "Mean Signal Intensity      : {:.3f}".format(mean_sig_intensity),
              "Peak Frequency             : {:.4f} Hz".format(peak_freq),
              "STD of Detrended Signal    : {:.3f}".format(sd_roi),
              "SNR                        : {:.3f}".format(SNR),
              "TFN                        : {:.3f}".format(aver_TFN),
              "SFNR                       : {:.3f}".format(aver_SFNR),
              "tSNR                       : {:.3f}".format(aver_tSNR),
              "Percent Fluctuation        : {:.3f} %".format(percent_fluc),
              "Drift                      : {:.3f} %".format(percent_drift),
              "Drift Fit                  : {:.3f} %".format(percent_drift_fit),
              "RDC                        : {:.3f} pixel".format(RDC),
              "Spectrum MAD               : {:.3f}".format(max_spec_outlier),
              "Ghost Rate                 : {:.3f} %".format(ghost_rate),
              "Background Noise Rate      : {:.3f} %".format(mean_background_rate),
              '',
              "\n Notice",
              "DIFF = Difference between Odd-volumes and Even-volumes",
              "SNR = Signal-to-Noise Ratio",
              "    = (mean signal)/(standard deviation of DIFF)",
              "TFN = standard deviation of detrended signal",
              "SFNR = Signal-to-Fluctuation-Noise Ratio",
              "     = (mean signal)/(standard deviation of detrended signal)",
              "tSNR = temporal Signal-to-Noise Ratio",
              "     = (mean signal)/(standard deviation of signal)",
              "Percent Fluctuation = (STD of residuals of 2nd-fit-time-series)/(mean signal)",
              "Drift = (max - min)/(mean signal)",
              "Drift Fit = (maxfit - minfit)/(mean signal)",
              "CV = Coefficient of Variance"
              "RDC = Radius of DeCorrelation, CV(1)/CV(max)",
              "MAD = Median Absolute Deviation = (max)/(median of Absolute Deviation)",
              "Ghost Rate = (mean ghost)/(mean signal)",
              "Background Rate = (mean background)/(mean signal)"
              ]
for text in text_content:
    y = y - 15
    pdf.drawString(80,y, text)
pdf.save()

# 4.2.2 Map
figure1_direction = output_dir + '_map.pdf'
fig, axes = plt.subplots(3, 2, figsize=(10, 12))
# Mean Signal Image
rect = Rectangle((ROI_y.start, ROI_x.start), roi_width, roi_width, linewidth=2, edgecolor='y', facecolor='none')
img_time_aver_image = axes[0,0].imshow(time_aver_image,origin='lower',cmap='gray')
fig.colorbar(img_time_aver_image,ax=axes[0,0])
axes[0,0].set_title('Mean Signal Image')
axes[0,0].add_patch(rect)
# DIFF
img_DIFF = axes[0,1].imshow(DIFF,origin='lower')
fig.colorbar(img_DIFF,ax=axes[0,1])
axes[0,1].set_title('SSN (DIFF) Map')
# SNR
img_SNR_full = axes[1,0].imshow(SNR_full,origin='lower')
fig.colorbar(img_SNR_full,ax=axes[1,0])
axes[1,0].set_title('SNR Map')
# TFN
img_TFN_full = axes[1,1].imshow(TFN_full,origin='lower')
fig.colorbar(img_TFN_full,ax=axes[1,1])
axes[1,1].set_title('TFN Map')
# SFNR
img_SFNR_full = axes[2,0].imshow(SFNR_full,origin='lower')
fig.colorbar(img_SFNR_full,ax=axes[2,0])
axes[2,0].set_title('SFNR Map')
# tSNR
img_tSNR = axes[2,1].imshow(tSNR_full,origin='lower')
fig.colorbar(img_tSNR,ax=axes[2,1])
axes[2,1].set_title('tSNR Map')
plt.savefig(figure1_direction)
plt.close()

# 4.2.3 Plot
figure2_direction = output_dir + '_plot.pdf'
fig, axes = plt.subplots(7,figsize=(10, 16))
# Mean Signal Intensity Time-Series
axes[0].plot(mean_sig_intensity_t,label='Mean signal intensity')
axes[0].plot(y_fit,color='r',label='Fit')
axes[0].set_xlabel("Volume Number")
axes[0].set_ylabel("Intensity")
axes[0].set_title('Mean Signal Intensity Time-Series')
axes[0].legend()
axes[0].grid(True, linestyle='--')
# Detrended Mean Signal Intensity Time-Series
axes[1].plot(mean_sig_intensity_t_detrend)
axes[1].set_xlabel("Volume Number")
axes[1].set_ylabel("Intensity")
axes[1].set_title('Detrended Mean Signal Intensity Time-Series')
axes[1].grid(True, linestyle='--')
# Fluctuation Frequency Spectrum
axes[2].plot(freq,spec)
axes[2].set_xlabel("Frequency (Hz)")
axes[2].set_ylabel("Magnitude")
axes[2].set_title('Fluctuation Frequency Spectrum')
axes[2].grid(True, linestyle='--')
# Distribution of the Signal Fluctuation
bins = 20
axes[3].hist(mean_sig_intensity_t_detrend, bins=bins, rwidth=0.8, density=True)
axes[3].plot(norm_x, norm_y, 'r-', label='Normal Distribution Fit')
axes[3].set_xlabel("Signal intensity")
axes[3].set_ylabel("Probability Density")
axes[3].set_title('Distribution of the Signal Fluctuation')
axes[3].legend()
axes[3].grid(True, linestyle='--')
# Coefficient of Variant
roi_list = range(1,roi_width+1)
axes[4].plot(roi_list,CV*100,linewidth=1,marker='.',markersize=10,color='b',label='Measured')
axes[4].plot(roi_list,CV_ideal*100,linewidth=1,marker='.',markersize=10,color='r',label='Theoretical')
axes[4].set_xlabel("ln(ROI width pixels)")
axes[4].set_ylabel("ln(100*CV)")
axes[4].set_title('Coefficient of Variant')
axes[4].legend()
axes[4].set_yscale('log')
axes[4].set_xscale('log')
axes[4].grid(True, linestyle='--')
# Ghost Rate Time-Series
axes[5].plot(ghost_rate_t,color='b')
axes[5].set_xlabel("Volume Number")
axes[5].set_ylabel("Ghost Rate (%)")
axes[5].set_title('Ghost Rate Time-Series')
axes[5].grid(True, linestyle='--')
# Background Intensity Time-Series
axes[6].plot(background_t,color='b')
axes[6].set_xlabel("Volume Number")
axes[6].set_ylabel("Background Intensity")
axes[6].set_title('Background Intensity Time-Series')
axes[6].grid(True, linestyle='--')
plt.savefig(figure2_direction)
plt.close()

# 4.2.4 Merge
output_direction = output_dir + '_Report.pdf'
pdf_files = [text_direction, figure1_direction, figure2_direction]
merger = PdfMerger()
for file in pdf_files:
    merger.append(file)
with open(output_direction, "wb") as output_file:
    merger.write(output_file)
    merger.close()
# Delete the useless pdf files
for file in pdf_files:
    os.unlink(file)
print('Success!')

