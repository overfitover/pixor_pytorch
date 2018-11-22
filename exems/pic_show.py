# import cv2   
import os
from PIL import Image
import matplotlib.pyplot as plt

  
# img = cv2.imread("./predict.png")   
# print(img)
# cv2.namedWindow("Image")   
# cv2.imshow("Image", img)   
# cv2.waitKey (0)  
# cv2.destroyAllWindows()  
current_path = os.path.split(os.path.realpath(__file__))[0]
img = Image.open(os.path.join(current_path, 'predict.png')) 
plt.figure("Image") 
plt.imshow(img) 
# plt.axis('on') 
# plt.title('image') 
plt.show()
