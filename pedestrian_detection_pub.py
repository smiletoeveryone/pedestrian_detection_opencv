import cv2
import imutils 

# Initializing the HOG person 
# detector 
hog = cv2.HOGDescriptor() 
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cap = cv2.VideoCapture('Pedestrians Compilation_c.mp4')

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
capture_size = (int(cap.get(3)), int(cap.get(4)))
out = cv2.VideoWriter('output.avi', fourcc, 20.0, capture_size)

while cap.isOpened(): 
    # Reading the video stream 
    ret, image = cap.read() 
    
    if ret: 
        
        image = cv2.flip(image,2)
        
   
        # Detecting all the regions  
        # in the Image that has a  
        # pedestrians inside it 
        (regions, _) = hog.detectMultiScale(image, 
                                            winStride=(4, 4), 
                                            padding=(4, 4), 
                                            scale=1.05) 
   
        # Drawing the regions in the  
        # Image 
        for (x, y, w, h) in regions: 
            cv2.rectangle(image, (x, y), 
                          (x + w, y + h),  
                          (0, 0, 255), 2) 
               
        # Showing the output Image
        out.write(image) 
        cv2.imshow("Image", image) 
        
        
        if cv2.waitKey(25) & 0xFF == ord('q'): 
            break
    else: 
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
