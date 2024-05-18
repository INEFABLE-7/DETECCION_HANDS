import cv2
import numpy as np

# Función para contar los dedos levantados
def contar_dedos(contorno, frame):
    hull = cv2.convexHull(contorno, returnPoints=False)
    if len(hull) > 3:
        defects = cv2.convexityDefects(contorno, hull)
        if defects is not None:
            count = 0
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(contorno[s][0])
                end = tuple(contorno[e][0])
                far = tuple(contorno[f][0])
                # Calcular los ángulos y distancias entre los puntos
                a = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                b = np.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                c = np.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                angle = np.arccos((b**2 + c**2 - a**2) / (2*b*c))
                # Si el ángulo es menor que 90 grados, contabilizar el dedo
                if angle <= np.pi / 2:
                    count += 1
                    # Dibujar líneas y puntos de interés
                    cv2.line(frame, start, end, [0, 255, 0], 2)
                    cv2.circle(frame, far, 5, [0, 0, 255], -1)
            return count
    return 0

# Configuración de la captura de video
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detección de manos
    hand_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'hand.xml')
    hands = hand_cascade.detectMultiScale(gray, 1.1, 5)

    # Procesar cada mano detectada
    for (x, y, w, h) in hands:
        # Recortar la región de la mano
        hand_region = frame[y:y+h, x:x+w]

        # Dibujar el rectángulo de detección
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Contar dedos levantados
        fingers = contar_dedos(contour, frame)

        # Mostrar resultado
        cv2.putText(frame, f'Dedos levantados: {fingers}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Mostrar el frame con las manos detectadas
    cv2.imshow('Hand Detection', frame)

    # Salir al presionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()

