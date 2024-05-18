import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Rangos de posiciones para una sonrisa
SMILING_RANGE_MIN = 0.3
SMILING_RANGE_MAX = 0.6

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5) as face_mesh:

    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        frame = cv2.flip(frame,1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks is not None:
            for face_landmarks in results.multi_face_landmarks:
                # Extraer las coordenadas de las esquinas de la boca
                mouth_left = face_landmarks.landmark[13]
                mouth_right = face_landmarks.landmark[14]
                mouth_left_x, mouth_left_y = mouth_left.x, mouth_left.y
                mouth_right_x, mouth_right_y = mouth_right.x, mouth_right.y
                
                # Calcular la distancia entre las esquinas de la boca
                mouth_distance = abs(mouth_right_x - mouth_left_x)
                
                # Determinar si la persona está sonriendo
                if SMILING_RANGE_MIN <= mouth_distance <= SMILING_RANGE_MAX:
                    cv2.putText(frame, "Feliz :)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Triste :(", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Dibujar los puntos clave faciales
                mp_drawing.draw_landmarks(frame, face_landmarks,
                    mp_face_mesh.FACEMESH_TESSELATION,
                    mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=1))

        cv2.imshow("Frame", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

cap.release()
cv2.destroyAllWindows()
