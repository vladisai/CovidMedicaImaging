import numpy as np
from skimage import feature as F
import cv2
from param import args 

def prepare_image(image,target_size=(224,224)):
  if len(image.shape)==3 and image.shape[0]==1:
    image=image[0]
  try:
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
  except:
    pass
  image=cv2.resize(image,target_size)
  try:
    image=np.divide(image-np.min(image),np.max(image)-np.min(image))*255
  except:
    pass
  return image

def get_lbp(image,P=8,R=1):
  image=prepare_image(image)
  lbp_image=F.local_binary_pattern(image,R=R,P=P)
  lbp_hist=np.histogram(lbp_image.reshape(-1),bins=256,range=(0,256))[0]
  return lbp_hist

def get_hog(image,comp_count=5000):
  image=prepare_image(image)
  pixs=int(np.sqrt(32.*np.prod(image.shape)/comp_count))
  hog_feature=F.hog(image,orientations=8,pixels_per_cell=(pixs,pixs),cells_per_block=(2,2),visualize=False,feature_vector=True)
  return np.array(hog_feature.reshape(-1))

def get_fft(image,comp_count=5000):
  image=prepare_image(image)
  # np.fft.restore_all()
  dft=np.fft.fft2(image)
  mag=np.asarray(20*np.log(np.abs(dft)))
  shifted=np.fft.fftshift(mag)
  dim=int(np.sqrt(comp_count))
  res=cv2.resize(shifted,(dim,dim))
  return res.reshape(-1)
