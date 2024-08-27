import numpy as np
import cv2
import matplotlib.pyplot as plt

def calculate_snr(image, noise):
    # Calculate signal power
    signal_power = np.mean(np.square(image))

    # Calculate noise power
    noise_power = np.mean(np.square(noise))

    # Calculate SNR
    snr = 10 * np.log10(signal_power / noise_power)
    
    return snr

def calculate_cnr(image, noise):
    # Calculate mean signal
    mean_signal = np.mean(image)

    # Calculate mean noise
    mean_noise = np.mean(noise)

    # Calculate standard deviation of signal
    std_signal = np.std(image)

    # Calculate standard deviation of noise
    std_noise = np.std(noise)

    # Calculate CNR
    cnr = np.abs(mean_signal - mean_noise) / np.sqrt(std_signal**2 + std_noise**2)

    return cnr

def calculate_psnr(image, reconstructed_image):
    # Convert images to float64
    image = np.float64(image)
    reconstructed_image = np.float64(reconstructed_image)

    # Calculate MSE
    mse = np.mean((image - reconstructed_image) ** 2)

    # Calculate maximum pixel value
    max_pixel_value = np.max(image)

    # Calculate PSNR
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))

    return psnr

def clahe(img, clip_limit=2.0, tile_size=(8, 8)):
    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    
    # Convert image to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # Split LAB channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to the L channel
    cl = clahe.apply(l)
    
    # Merge the CLAHE enhanced L channel with the original A and B channels
    limg = cv2.merge((cl, a, b))
    
    # Convert LAB image back to BGR
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def sharpen(image, sharpen_strength=1.0):
    # Applies Gausian blur to the image to create a smoothed version and then subtracts the blurred version to enhance the edges, -> sharpened image
    blurred = cv2.GaussianBlur(image, (0, 0), 3)
    sharpened = cv2.addWeighted(image, 1.0 + sharpen_strength, blurred, -sharpen_strength, 0)
    return sharpened

def denoise(image, strength=10):
    # Apply non-local means denoising
    denoised = cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)
    return denoised

def plot_histogram(image):
    # Calculate histogram
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    
    # Plot histogram
    plt.figure(figsize=(6, 4))
    plt.plot(hist, color='black')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.title('Histogram')
    plt.xlim([0, 256])
    plt.grid(True)
    plt.tight_layout()
    
    # Convert plot to numpy array
    plt_img = plt.gcf()
    plt_img.canvas.draw()
    img_np = np.array(plt_img.canvas.renderer.buffer_rgba())
    plt.close()
    
    return img_np[:, :, :3]