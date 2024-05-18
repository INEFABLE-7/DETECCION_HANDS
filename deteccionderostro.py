import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Índices de los puntos faciales que quieres resaltar
index_list = [70, 63, 105, 66, 107, 336, 296, 334, 293, 300,
                122, 196, 3, 51, 281, 248, 419, 351, 37, 0, 267]

# Inicializar el objeto FaceMesh
with mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

    # Inicializar la captura de video desde la cámara
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        # Leer un frame de la cámara
        ret, frame = cap.read()
        if not ret:
            break

        # Convertir el frame a RGB (Mediapipe requiere una imagen en RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Procesar el frame con FaceMesh
        results = face_mesh.process(frame_rgb)

        # Verificar si se detectaron landmarks faciales
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Dibujar los landmarks faciales en el frame
                for index in index_list:
                    x = int(face_landmarks.landmark[index].x * frame.shape[1])
                    y = int(face_landmarks.landmark[index].y * frame.shape[0])
                    cv2.circle(frame, (x, y), 2, (255, 0, 255), 2)

        # Mostrar el frame con los landmarks faciales
        cv2.imshow('Face Mesh', frame)

        # Salir si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar los recursos
    cap.release()
    cv2.destroyAllWindows()
