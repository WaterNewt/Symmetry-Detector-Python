import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

res = (720, 1280)
image = cv2.imread("photo.jpg")

with mp_face_detection.FaceDetection(
    model_selection=1, min_detection_confidence=0.5) as face_detection:

    while True:

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = face_detection.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(image, detection)
                bbox = detection.location_data.relative_bounding_box
                sym_score = detection.score[0] * 90
                cv2.putText(image, f'{sym_score:.2f}%', (int(bbox.xmin * image.shape[1]), int(bbox.ymin * image.shape[0]) + 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                image = cv2.resize(image, res)
                print(f'{sym_score:.2f}%')

        cv2.imshow('MediaPipe Face Detection', image)
        cv2.imwrite('output.jpg', image)
        if cv2.waitKey(0) & 0xFF == 27:
            break

cv2.destroyAllWindows()
