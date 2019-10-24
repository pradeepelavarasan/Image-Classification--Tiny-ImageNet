# Image-Classification-Tiny-ImageNet
Image Classification on Tiny ImageNet with Custom ML Model built based on RESNET and DENSENET. This was done for the External Internship Program EIP3 if The School of AI, Bangalore

## Steps involved:

### Importing Required Packages
Importing the required packages and also initializing the required processes. This also has the initialization if TPU process to leverage to power of TPU in Google Colabs. 
Note: There have been few updates after April 2019 (when this model was written) to the TPU execution. So kindly update to latest versio nof TPU execution code. 

### Loading data to Colab
1) Saved the image dataset in Google Drive
2) Mounted the Google Drive to Google Colab
3) Extracted the images to Google Colab

Above process is more efficient for using the images. 

### Custom Model Design based on RESNET and DENSENET

Considerations for the model design:
1) Size of the images
2) Based on the size of images, we had to decide the Receptive field required. Since the images are very small in size, we are had to go beyong the receptive field of the object to cover the background also. So we target RF of about 100.
3) Given the small size of data, carry forward the images to deeper layers in the network without reducing its size much.

Below is the high levelstructure of the custom model:

- Initialization Block 0
- Intersection/Bottleneck Block 0
- Resnet Block 1
- Intersection/Bottleneck Block 1
- Resnet Block 2
- Intersection/Bottleneck Block 2
- Resnet Block 3
- Intersection/Bottleneck Block 3
- Resnet Block 4
- Intersection/Bottleneck Block 4
- Global Average pooling
- Softmax

Below is the view of how the model is designed considering the Receptive field, image size and no. of parameters at each level. You can also refer to excel file (Model Design.xlsx) where this model was designed

![Model_Design_Plan](/Model%20Design%20Plan.JPG)

### Data Generator Setup

I used the ImageDataGenerator and flow_from_directory function to read the images from the directly on the fly during model execution. This approach is better than converting all the images into pixels and handling the huge file for processing. 
Another advantage of this approach is the ability to do data augmentation as part of the ImageDataGenerator. The augmentations perfomed are:
- Horizontal Flip
- Zoom in/ Zoom out
- rescalling
- Imgaug: ChannelShuffle
- Imgaug: CoarseDropout

The type of data augmentations required should be choosen based on analysing the actual images from the data set and also more importantly the misclasified images in the dataset. This will help us build the model not depending on just few indicators for a specific class. This is an iterative process and you can refine it after analysing the misclassified images. 

### Training Initializations

### Model Training on 32*32 Images

### Model Training on 64*64 images

### Additional Data Augmentations

### Model Training on 64*64 images - with Additional Augmentations

### Final Evaluation

The best accuracy reached was 59.49% in epoch 198
