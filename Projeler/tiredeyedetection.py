import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist

# Göz açıklık oranını hesaplamak için gerekli fonksiyon
def calculate_eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Yüz ve göz algılayıcıları tanımlama
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Kamera bağlantısını başlatma
cap = cv2.VideoCapture(0)

# Göz açıklık eşik değeri ve ardışık kapanan kare sayısı
EYE_AR_THRESH = 0.2  # Eşik değeri düşürüldü
EYE_AR_CONSEC_FRAMES = 3  # Düşük bir değer seçildi, örneğin 3

COUNTER = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    for face in faces:
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (102, 0, 153), 3)
        
        landmarks = predictor(gray, face)
        
        left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
        right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])
        
        for eye in left_eye:
            cv2.circle(frame, tuple(eye), 2, (0, 255, 0), -1)
            
        for eye in right_eye:
            cv2.circle(frame, tuple(eye), 2, (0, 255, 0), -1)
        
        left_ear = calculate_eye_aspect_ratio(left_eye)
        right_ear = calculate_eye_aspect_ratio(right_eye)
        
        ear = (left_ear + right_ear) / 2.0
        
        if ear < EYE_AR_THRESH:
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                cv2.putText(frame, "YORGUNLUK ALGILANDI!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            COUNTER = 0
    
    cv2.imshow("Frame", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
