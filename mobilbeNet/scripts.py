import cv2
import numpy as np

# Chemin vers les fichies téléchargés
prototxt = './deploy.prototxt' # Chemin vers le fichier prototxt
caffe_model =  './mobilenet.caffemodel' # Chemin vers le modéle pré-entraîné

# Charger le réseau de détection
net = cv2.dnn.readNetFromCaffe(prototxt,caffe_model)

class_names = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'dog', 'horse', 'motorbike', 'person', 'sheep', 'sofa', 'train', 'tvmonitor']

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Préparer l'image pour la détection d'objects
    h,w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), (127.5, 127.5, 127.5), swapRB=False, crop=False)
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
       
        # On détecte l'objet seulement si la confiance est suffisante
        if confidence > 1.2:
            idx = int(detections[0, 0, i, 1])

            # # Si l'object détecté est une personne (index 15 pour "person" dans la liste)
            # if class_names[idx] == 'person':
            #     box = detections[0, 0, i, 3:7] * np.array([w,h,w,h])
            #     (startX, startY, endX, endY) = box.astype("int")

            #     # Masquer la personne par l'arrière-plan ou une zone transparente
            #     # frame[startY:endY, startX:endX] = frame[startY:endY, startX:endX] * 0  # Rendre la zone noir (invisisble)
            #     print("hanif")
            #+++++++++++++++++++++++++++++++++++

            #On récupère la classe de l'object détecté
            object_name = class_names[idx]

            # Récupérer les coordonnées du rectangle autour de l'object détecter 
            box = detections[0, 0, i, 3:7] * np.array([w,h,w,h])
            (startX, startY, endX, endY) = box.astype('int')
            
            # Afficher le nom de l'object sur l'image
            label = f"{object_name}: {confidence:.2f}"
            cv2.putText(frame, label,(startX, startY -10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 0, 0), 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)
    # Affichage de l'image résultante
    cv2.imshow('Objet Detection', frame)

    # Sortie sur la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Libérer la capture et fermer les fenêtrs
cap.release()
cv2.destroyAllWindows()
    # # Affichage de l'image résultante
    # cv2.imshow('Invisible Person Effect', frame)

    # # Sortie sur la touche 'q'
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# Libérer la capture et fermer les fenêtres
# cap.release()
# cv2.destroyAllWindows()