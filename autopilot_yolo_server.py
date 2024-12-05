from picamera2 import Picamera2
import torch
import cv2
import YB_Pcb_Car as car
import time
import numpy as np
import threading
from tensorflow import keras
import requests

best = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/pi/best.pt', force_reload=True)

image_path = 'image.jpeg'
image_url = 'http://15.165.162.155:8080/attackraspi'

def send_image(image_path, image_url):
    try:
        with open(image_path, 'rb') as img_file:
            files = {'image': ('image.jpeg', img_file, 'image/jpeg')}
            response = requests.post(image_url, files=files)
        return response
    except requests.RequestException as e:
        print(f'Error sending image: {e}')
        return None
    
def send_misrecognized_sign(adversarial_label, original_label):
    result_url = "http://43.202.104.135:3000/errors/info"
    data = {
        "misrecognized_sign_name": adversarial_label,
        "recognized_sign_name": original_label
    }
    try:
        response = requests.post(result_url, json=data)
        if response.status_code in (200, 201):
            print("Success: Sign name sent successfully.\n", response.json())
        else:
            print("Error generated", response.json())
    except Exception as e:
        print("Error generated in requesting:", e)

def control_car(car_instance, direction, speed_left, speed_right):
    if direction == "left":
        car_instance.Car_Left(speed_left, speed_right)
    elif direction == "right":
        car_instance.Car_Right(speed_left, speed_right)
    elif direction == "run":
        car_instance.Car_Run(speed_left, speed_right)
    elif direction == "stop":
        car_instance.Car_Stop()

def img_preprocess(image):
    height, _, _ = image.shape
    image = image[int(height/4):,:,:]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    image = cv2.resize(image, (320, 180)) 
    image = image / 255
    return image

def main():
    car_instance = car.YB_Pcb_Car()
    picam2 = Picamera2()
    picam2.preview_configuration.main.size = (320, 240)
    picam2.preview_configuration.main.format = 'RGB888'
    picam2.preview_configuration.align()
    picam2.configure("preview")
    picam2.start()

    try:
        model = keras.models.load_model('/home/pi/AI_CAR/model/lane_navigation_final.keras')
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
    
    while True:
        image = picam2.capture_array()
        results = best(image)

        if len(results.xyxy[0]) > 0:
            result = results.xyxy[0][0]
            x1, y1, x2, y2 = map(int, result[:4])

            cropped_image = image[y1:y2, x1:x2]
            enlarged_image = cv2.resize(cropped_image, (480, 480))
            cv2.imwrite('image.jpeg', enlarged_image)

            response = send_image(image_path, image_url)

            if response is not None and response.status_code in (200, 201):
                try:
                    response_data = response.json()
                    adversarial_label = response_data.get("adversarial_label", "N/A")
                    original_label = response_data.get("original_label", "N/A")
                    print(f'success: adversarial_label={adversarial_label}, original_label={original_label}\n')
                    
                    if adversarial_label  == "notLeft":
                        print("Adversarial label detected: notLeft, car  stopping")
                        control_car(car_instance, "stop", 0, 0)
                        time.sleep(3)

                    elif adversarial_label  == "notEnter":
                        print("Adversarial label detected: stop, car stopping")
                        control_car(car_instance, "stop", 0, 0)
                        time.sleep(3)

                    elif adversarial_label  == "slow":
                        print("Adversarial label detected: slow, car slowing down")
                        control_car(car_instance, "run", 30, 30)
                        time.sleep(1)
                        control_car(car_instance, "stop", 0, 0)
                        time.sleep(3)

                    elif adversarial_label  == "right":
                        print("Adversarial label detected: right, car right")
                        control_car(car_instance, "right", 40, 0)
                        time.sleep(1)
                        control_car(car_instance, "stop", 0, 0)
                        time.sleep(3)

                    misrecognized_response = send_misrecognized_sign(adversarial_label, original_label)
                    if misrecognized_response is not None:
                        print('Adversarial label sent successfully')
                
                except ValueError:
                    print('success: server respond no JSON form')
    
            else:
                print(f'fail : {response.status_code if response else "No response"}')

            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)  

        cv2.imshow('YOLO Detection', image)

        if image is None:
            print("Failed to grab frame")
            time.sleep(1)  
            continue
        
        print("Frame grabbed successfully")
        #cv2.imshow('Original', image)
        
        preprocessed = img_preprocess(image)
        #cv2.imshow('pre', preprocessed)
        X = np.asarray([preprocessed])
        try:
            steering_angle = int(model.predict(X)[0])
            print("Predicted angle:", steering_angle)
        except Exception as e:
            print(f"Error during prediction: {e}")
            continue
        
        if -1<= steering_angle <= 102:
            print("go")
            threading.Thread(target=control_car, args=(car_instance, "run", 30, 30)).start()
        elif 102< steering_angle :
            print("right")
            threading.Thread(target=control_car, args=(car_instance, "right", 50, 0)).start()
        elif steering_angle < -1: 
            print("left")
            threading.Thread(target=control_car, args=(car_instance, "left", 0, 50)).start()
        else:
            print("stop")
            threading.Thread(target=control_car, args=(car_instance, "stop", 0, 0)).start()

        time.sleep(0.1) 

        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()
    print("Exited main loop")

if __name__ == '__main__':
    main()
    