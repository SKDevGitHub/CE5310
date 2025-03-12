import cv2

#CAPTURES A FRAME TO USE FOR DATA
def capture_frame(video):
    ret, frame = video.read()
    if ret:
        return frame
    return None

#ADDS BOUNDING BOXES TO THE FRAME AND DISPLAYS
def display_frame(frame, detect):
    if frame is None:
        return
    
    #DRAWS BOXES AROUND EACH FOUND ITEM
    for detection in detect:
        box = detection["box"]
        object_type = detection["class"]
        x1,y1,x2,y2 = box
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0, 255, 0), 2)
        cv2.putText(frame, object_type, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
    cv2.imshow("Live Feed", frame)
    
    #QUIT TIME
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return False
    return True

#OPENS CAMERA
def open_camera(camera_index = 0):
    return cv2.VideoCapture(camera_index)

#CLOSE CAMERA
def close_camera(capture):
    capture.release()
    cv2.destroyAllWindows()