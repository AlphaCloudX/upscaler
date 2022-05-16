import cv2
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

img1 = cv2.imread(r'.\data\runs\run baseModel 100epoch 256x 2x\runsaiScaled_2.jpg')
img2 = cv2.imread(r'.\data\runsaiScaled_2.jpg')
img3 = cv2.imread(r'.\data\testPexel\highRes\pexels-eberhard-grossgasteiger-1699030.jpg')

MSE1 = mean_squared_error(img1, img3)
MSE2 = mean_squared_error(img2, img3)

PSNR1 = peak_signal_noise_ratio(img1, img3)
PSNR2 = peak_signal_noise_ratio(img2, img3)

SSIM1 = structural_similarity(img1, img3, multichannel=True)
SSIM2 = structural_similarity(img2, img3, multichannel=True)

print('MSE BaseModel vs Original: ', MSE1)
print('MSE newModel vs Original: ', MSE2)
print("")
print('PSNR BaseModel vs Original: ', PSNR1)
print('PSNR newModel vs Original: ', PSNR2)
print("")
print('SSIM BaseModel vs Original: ', SSIM1)
print('SSIM newModel vs Original: ', SSIM2)