import numpy as np
from skimage import feature as F
import cv2

def prepare_image(image,target_size=(224,224)):
  try:
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
  except:
    pass
  image=cv2.resize(image,target_size)
  image=np.divide(image+1024,2048)*255
  return image

def get_lbp(image,P=8,R=1):
  image=prepare_image(image)
  lbp_image=F.local_binary_pattern(image,R=R,P=P)
  lbp_hist=np.histogram(lbp_image.reshape(-1),bins=256,range=(0,256))[0]
  return lbp_hist

def get_hog(image):
  image=prepare_image(image)
  hog_feature=F.hog(image,orientations=9,pixels_per_cell=(8,8),visualize=False,feature_vector=True)
  return np.array(hog_feature.reshape(-1))

def get_fft(image):
  image=prepare_image(image)
  np.fft.restore_all()
  dft=np.fft.fft2(image)
  mag=np.asarray(20*np.log(np.abs(dft)))
  shifted=np.fft.fftshift(mag)
  return shifted.reshape(-1)
