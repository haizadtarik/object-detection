import numpy as np
import cv2
import time
from keras.preprocessing import image
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input, decode_predictions

#----------- CNN Model ---------
model = MobileNet(weights='imagenet')

# ---------- IP Cam URL ------
cap = cv2.VideoCapture('http://xxx.xxx.x.x:xxxx/video')

while(True):
    start_time = time.time()
    
    ret, frame = cap.read()
    
    # Preprocess input
    img = cv2.resize(frame, (224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    # Run object detection
    preds = model.predict(x)
    
    # Decode prediction 
    prediction = decode_predictions(preds, top=1)[0]
    
    # Dislpay FPS
    cv2.putText(frame, 'FPS: '+ str(1/(time.time()-start_time)), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    # Display output
    cv2.putText(frame, 'Object: '+ str(prediction[0][1]) + ',' + ' Confidence = ' + str(prediction[0][2]), (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    # Dislpay FPS
    cv2.putText(frame, 'Press Q to quit', (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    
    # Display video stream
    cv2.imshow('frame',frame)
    
    # Reduce stress to processor
    time.sleep(0.1) 

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

cap.release()
cv2.destroyAllWindows()
