import cv2
import YB_Pcb_Car as car
import torch
import threading
from picamera2 import Picamera2
import requests
import time

best = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/pi/best.pt', force_reload=True)

image_path = 'image.jpeg'
image_url = 'http://15.165.162.155:8080/attackraspi'

manual_mode = False
last_sent_time = 0  
send_interval = 8 
last_detection_time = 0


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

def process_sign_detection(car_instance, adversarial_label):
    global manual_mode
    
    if adversarial_label == "notLeft" or adversarial_label == "notEnter":
        print(f"Detected '{adversarial_label}', stopping the car.")
        control_car(car_instance, "stop", 0, 0)

    elif adversarial_label == "slow":
        print("Detected 'slow', car slowing down.")
        control_car(car_instance, "stop", 0, 0)

    elif adversarial_label == "right":
        print("Detected 'right', turning car right.")
        control_car(car_instance, "stop", 0, 0)

    manual_mode = False

def control_car(car_instance, direction, speed_left, speed_right):
    if direction == "left":
        car_instance.Car_Left(speed_left, speed_right)
    elif direction == "right":
        car_instance.Car_Right(speed_left, speed_right)
    elif direction == "run":
        car_instance.Car_Run(speed_left, speed_right)
    elif direction == "back":
        car_instance.Car_Back(speed_left, speed_right)
    elif direction == "stop":
        car_instance.Car_Stop()

def main():
    global manual_mode,last_sent_time,last_detection_time 

    car_instance = car.YB_Pcb_Car()
    picam2 = Picamera2()
    picam2.preview_configuration.main.size = (320, 320)
    picam2.preview_configuration.main.format = 'RGB888'
    picam2.preview_configuration.align()
    picam2.configure("preview")
    picam2.start()
   
    #carState = "stop"
    while True:
        frame = picam2.capture_array()
        results = best(frame)

        if len(results.xyxy[0]) > 0:
            result = results.xyxy[0][0]
            x1, y1, x2, y2 = map(int, result[:4])

            cropped_image = frame[y1:y2, x1:x2]
            enlarged_image = cv2.resize(cropped_image, (480, 480))
            cv2.imwrite('image.jpeg', enlarged_image)

            #response = send_image(image_path, image_url)
            if time.time() - last_detection_time >= send_interval:
                response = send_image(image_path, image_url)
                last_detection_time = time.time()

                if response is not None and response.status_code in (200, 201):
                    try:
                        response_data = response.json()
                        adversarial_label = response_data.get("adversarial_label", "N/A")
                        original_label = response_data.get("original_label", "N/A")
                        print(f'success: adversarial_label={adversarial_label}, original_label={original_label}\n')
                        threading.Thread(target=process_sign_detection, args=(car_instance, adversarial_label)).start()
                    
                        misrecognized_response = send_misrecognized_sign(adversarial_label, original_label)
                        if misrecognized_response is not None:
                            print('Adversarial label sent successfully')
                    
                    except ValueError:
                        print('success: server respond no JSON form')
                else:
                    print(f'fail : {response.status_code if response else "No response"}')

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  
            manual_mode = True
        else:
            manual_mode = True       
        cv2.imshow('YOLO Detection', frame)

        keyValue = cv2.waitKey(1) & 0xFF
        if keyValue == ord('q'):  
            break
        elif keyValue == 82:  
            print("go")
            manual_mode = True
            threading.Thread(target=control_car, args=(car_instance, "run", 40, 40)).start()
        elif keyValue == 84:  
            print("back")
            manual_mode = True
            threading.Thread(target=control_car, args=(car_instance, "back", 30, 30)).start()
        elif keyValue == 81:  
            print("left")
            manual_mode = True
            threading.Thread(target=control_car, args=(car_instance, "left", 0, 50)).start()
        elif keyValue == 83:  
            print("right")
            manual_mode = True
            threading.Thread(target=control_car, args=(car_instance, "right", 50, 0)).start()
        elif keyValue == ord('s'):  
            print("stop")
            manual_mode = True
            threading.Thread(target=control_car, args=(car_instance, "stop", 0, 0)).start()
        #elif keyValue == ord('a'): 
        #    print("Switching back to automatic mode")
        #    manual_mode = False
    cv2.destroyAllWindows()
    picam2.stop()

if __name__ == '__main__':
    main()