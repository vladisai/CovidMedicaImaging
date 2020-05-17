import numpy as np
from os.path import isfile,join
from skimage import feature as F
from os import listdir
import logging
import argparse
import cv2
import os

def save_data(output_path,data):
  features,images=data
  for key in features:
    np.save(output_path+'/features_'+key+'.npy',features[key])
    # saves one sample image for each feature
    cv2.imwrite(output_path+'/image_'+key+'.png',images[key][0])

def load_data(input_path,target_size=None,crop_by_pixels=None,crop_by_factors=None,**kwargs):
  try:
    img_names=[img_name for img_name in listdir(input_path) if (isfile(join(input_path,img_name)) and not img_name.startswith('.'))]
    logging.info(f'Number of files read: {len(img_names)}')
    if len(img_names)<1:
      raise ValueError("no images read")
  except:
    raise ValueError("input_path dir not found / could not read")
  images=[cv2.imread(join(input_path,img_name)) for img_name in img_names]
  try:
    images=list(map(lambda x: cv2.cvtColor(x,cv2.COLOR_BGR2GRAY),images))
    logging.info('Images converted to greyscale')
  except:
    logging.info('Images are greyscale')
  target_size=target_size or images[0].shape
  images=list(map(lambda x: cv2.resize(x,target_size),images))
  if crop_by_pixels:
    crop=(crop_by_pixels[1],crop_by_pixels[0])
    original=images[0].shape
    if images[0].shape[0]<crop[0] or images[0].shape[1]<crop[1]:
      raise AssertionError("crop size is larger than image itself")
    images=list(map(lambda x: x[int((x.shape[0]-crop[0])/2):-int((x.shape[0]-crop[0])/2),int((x.shape[1]-crop[1])/2):-int((x.shape[1]-crop[1])/2)],images))
    logging.info(f'Cropped by pixels: {crop}; original: {original}; new dimensions: {images[0].shape}')
  elif crop_by_factors:
    original=images[0].shape
    crop=(crop_by_factors[1],crop_by_factors[0])
    if crop[0]>1 or crop[0]<0 or crop[1]>1 or crop[1]<0:
      raise AssertionError("crop factors must be between 0 and 1")
    images=list(map(lambda x: x[int(x.shape[0]*(1-crop[0])/2):x.shape[0]-int(x.shape[0]*(1-crop[0])/2),int(x.shape[1]*(1-crop[1])/2):x.shape[1]-int(x.shape[1]*(1-crop[1])/2)],images))
    logging.info(f'Cropped by factors: {crop}; original: {original}; new dimensions: {images[0].shape}')
  return images

def get_lbp(images,P=8,R=1,**kwargs):
  lbp_images=list(map(lambda x: F.local_binary_pattern(x,R=R,P=P),images))
  lbp_hists=list(map(lambda x: np.histogram(x.reshape(-1),bins=256,range=(0,256))[0],lbp_images))
  return lbp_hists,lbp_images

def get_hog(images,visualize=True,dim_reduction_factor=1.0,bins=9,cells_per_block=(2,2),**kwargs):
  hog_features,hog_images=[],[]
  pixs=int(np.sqrt(np.prod(cells_per_block)*bins/dim_reduction_factor))
  for img in images:
    if visualize:
      hog_feature,hog_image=F.hog(img,orientations=bins,pixels_per_cell=(pixs,pixs),cells_per_block=cells_per_block,visualize=visualize,feature_vector=True)
      hog_images.append(hog_image)
    else:
      hog_feature=F.hog(img,orientations=bins,pixels_per_cell=(pixs,pixs),cells_per_block=cells_per_block,visualize=visualize,feature_vector=True)
    hog_features.append(hog_feature)
  if visualize:
    # brightening the image by ensuring that overall brightness is 10%.
    brightning_factor=25.5/np.mean(np.array(hog_images).reshape(-1))
    hog_images=list(map(lambda x: x*brightning_factor,hog_images))
    return hog_features,hog_images
  else:
    return hog_features

def get_fourier(images):
  dfts=[np.fft.fft2(x) for x in images]
  mags=np.asarray([20*np.log(np.abs(x)) for x in dfts],dtype=np.uint8)
  shifted=[np.fft.fftshift(x) for x in mags]
  return [np.divide(ex,np.max(ex)) for ex in shifted],shifted

def main(input_path,output_path,fft,lbp,hog,lbp_kwargs={},hog_kwargs={},preprocessing_kwargs={}):
  logging.getLogger().setLevel(logging.INFO)
  data=load_data(input_path,**preprocessing_kwargs)
  features={}
  images={}
  if fft:
    fft_features,fft_images=get_fourier(data)
    images['fft']=fft_images
    features['fft']=fft_features
    logging.info("Computed fft")
  if lbp:
    lbp_features,lbp_images=get_lbp(data,**lbp_kwargs)
    features['lbp']=lbp_features
    images['lbp']=lbp_images
    logging.info("Computed lbp")
  if hog:
    hog_features=get_hog(data,**hog_kwargs)
    if len(hog_features)>1:
      hog_images=hog_features[1]
      hog_features=hog_features[0]
      images['hog']=hog_images
    features['hog']=hog_features
    logging.info("Computed hog")
  data=[features,images]
  save_data(output_path,data)
  logging.info(f"Files written to '{output_path}'")

if __name__=='__main__':
  parser=argparse.ArgumentParser()
  parser.add_argument('--input_path',type=str,default=None)
  parser.add_argument('--output_path',type=str,default=None)
  parser.add_argument('--fft',type=int,default=1) # flag to compute fourier transform
  parser.add_argument('--lbp',type=int,default=1) # flag to compute local binary patterns
  parser.add_argument('--hog',type=int,default=1) # flag to compute histogram of gradients
  parser.add_argument('--lbp_kwargs',type=str,default='{}') # supports 'R':int (radius) and 'P':int (number of points)
  parser.add_argument('--hog_kwargs',type=str,default='{}') # supports 'bins':int (angle bins for the histogram), 
  # 'visualize':bool (compute images or features only), 'reduction_factor':float (hog transform reduces the dimensionality of examples by this constant)
  # 'cells_per_block': tuple(int,int) (dimensions of each normalization block. higher values increase overlapping and, hence, the final dimensionality)
  parser.add_argument('--preprocessing_kwargs',type=str,default='{}') # supports 'target_size':tuple(int,int) (resize all images before featurization)
  # 'crop_by_pixels':tuple(int,int) (crop the central piece of given height and width out of all images), 'crop_by_factors':tuple(float,float):
  # crops the central part of each image leaving only the given fractions of the original dimensions. note that cropping (if any) is applied after resizing (if any).
  args=parser.parse_args()
  main(args.input_path,args.output_path,args.fft,args.lbp,args.hog,eval(args.lbp_kwargs),eval(args.hog_kwargs),eval(args.preprocessing_kwargs))
