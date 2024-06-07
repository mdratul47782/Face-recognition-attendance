import cv2

cam_port = 0
cam = cv2.VideoCapture(cam_port)

# Reading the input using the camera
inp = input('Enter person name: ')

# If the image is detected without any error, show result
while True:
    result, image = cam.read()
    if result:
        cv2.imshow(inp, image)
        if cv2.waitKey(1) & 0xFF == ord('c'):  # Wait for 'c' key to capture image
            cv2.imwrite(inp + ".png", image)
            print("Image taken")
            break
    else:
        print("No image detected. Please try again.")
        break

# Release the camera and close all windows
cam.release()
cv2.destroyAllWindows()
