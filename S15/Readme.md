File description

## Raw images contains persons with hardhats, masks, vest and boots
## The images were annotaed accordingly with the help of yolo v3 annotation  - leading upto 4 classes of objects per image 
## Depth estimator model MiDas pretrained was used to draw inference and give the depth estimation in cmap
## PlaneRcNN was used to get the segmentations file on the same image as well the depth estimation

1. Total number of images : 3590
2. Depth Map is generated by Intel-MiDas
3. Bounding box is created using Planercnn - file type .png
4. Surface Box is created using  Planercnn - file type .npy 