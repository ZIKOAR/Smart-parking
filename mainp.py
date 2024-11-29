import cv2
import math
import easyocr
import webcolors
import pandas as pd
from ultralytics import YOLO
import cvzone
import numpy as np
import pytesseract
from datetime import datetime
import os
NBM = 10
places = [False] * NBM
#fonctions de detection du couleur
"""
def determine_color(h, s, v):
    if v <= 30:
        return "Black"
    elif v >= 225 and s <= 35:
        return "White"
    elif s < 50 and 30 < v < 225:
        return "Gray"
    elif 0 <= h < 10 or h >= 170:
        if s >= 100 and v >= 100:
            return "Red"
    elif 10 <= h < 30:
        if s >= 100 and v >= 100:
            return "Orange"
    
    elif 30 <= h < 90:
        if s >= 100 and v >= 100:
            return "Yellow"
    
    elif 90 <= h < 150:
        if s >= 100 and v >= 100:
            return "Green"
    
    elif 150 <= h < 200:
        if s >= 100 and v >= 100:
            return "Cyan"
    
    elif 200 <= h < 270:
        if s >= 100 and v >= 100:
            return "Blue"
    
    elif 270 <= h < 330:
        if s >= 100 and v >= 100:
            return "Violet"
    return "Unknown"
def find_closest_color(r, g, b):
    # Obtenir les couleurs nommées CSS3 et leurs codes hexadécimaux
    css3_colors = webcolors.CSS3_NAMES_TO_HEX
    
    # Convertir chaque couleur en RGB et créer un dictionnaire
    named_colors = {webcolors.hex_to_rgb(value): name.lower() for name, value in css3_colors.items()}
    
    # Préparer la couleur d'entrée
    input_color_rgb = np.array([r, g, b])
    
    closest_color_name = None
    min_distance = float('inf')
    
    # Trouver la couleur la plus proche
    for rgb, name in named_colors.items():
        distance = np.linalg.norm(np.array(rgb) - input_color_rgb)  # Calcul de la distance euclidienne
        if distance < min_distance:
            min_distance = distance
            closest_color_name = name
            
    return closest_color_name
def color_detection1(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    height, width, _ = frame.shape
    rect_center_x = int(width / 2)+60
    rect_center_y = int(height / 2)+40
    rect_width = 400
    rect_height = 400     
    x_min = max(0, rect_center_x - int(rect_width / 2))
    x_max = min(width, rect_center_x + int(rect_width / 2))
    y_min = max(0, rect_center_y - int(rect_height / 2))
    y_max = min(height, rect_center_y + int(rect_height / 2)) 
    h = np.mean(hsv_frame[y_min:y_max, x_min:x_max, 0])
    s = np.mean(hsv_frame[y_min:y_max, x_min:x_max, 1])
    v = np.mean(hsv_frame[y_min:y_max, x_min:x_max, 2])
    color=find_closest_color(h, s, v) 
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 255, 255), 2)
    cv2.putText(frame, color, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return color
def color_detection(frame):
    model = YOLO('yolov8n.pt')
    results = model.track(frame, persist=True)
    x1=x2=y1=y2=None
    image=frame
    for result in results:
        box = result.boxes.xyxy[0]  # Prendre la première boîte de détection
        x1, y1, x2, y2 = box.tolist() 
        print(x1,x2,y1,y2)
    car_image = image[int(y1):int(y2), int(x1):int(x2)]
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h = np.mean(hsv_frame[int(y1):int(y2), int(x1):int(x2), 0])
    s = np.mean(hsv_frame[int(y1):int(y2), int(x1):int(x2), 1])
    v = np.mean(hsv_frame[int(y1):int(y2), int(x1):int(x2), 2])
    color=find_closest_color(v, s, h) 

    #cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
    #cv2.putText(frame, color, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return color
"""
#marque detection
names1 = ['Mercedes', 'Hyundai', 'Suzuki', 'Nissan', 'Toyota', 'Mitsubishi', 'Ford', 'Volkswagen', 'Audi', 'BMW']
def marque_detection(frame):
    model= YOLO('best1.pt')
    results = model.predict(frame)
    class_name = "inconnu"
    confidence = 0.0
    if results:
        for r in results:
            for box in r.boxes:
                cls = box.cls[0]
                clsIndex = int(cls)
                class_name = names1[clsIndex]
                confidence = math.floor(box.conf[0] * 100) / 100

    return class_name, confidence 
#class des infos
class myclass:
    def __init__(self, numero,nomplaque, precision,marque,marque_prcision,couleur, datetime):
        self.numero = numero
        self.nomplaque = nomplaque
        self.precision =precision
        self.marque=marque
        self.marque_prcision=marque_prcision
        self.couleur =couleur
        self.datetime = datetime
def indice_value(tableau,value):
    for indice, element in enumerate(tableau):
        if element is value:
            return indice
def indice_text(tableau, value):
    for indice, element in enumerate(tableau):
        if getattr(element, 'nomplaque') == value :
            return indice
def prix_f(t1_str,t2_str):
    t1 = datetime.strptime(t1_str, "%Y-%m-%d %H:%M:%S")
    t2 = datetime.strptime(t2_str, "%Y-%m-%d %H:%M:%S")
    heures = (t2 - t1).total_seconds() / 3600 
    return heures*2
def efface(nom_fichier, texte_recherche):
    with open(nom_fichier, "r") as file_in:
        lignes = file_in.readlines()
    with open("temp.txt", "w") as file_out:
        for ligne in lignes:
            if texte_recherche not in ligne:
                file_out.write(ligne)
    os.replace("temp.txt", nom_fichier)

#model de detection du plaque
model = YOLO('best.pt')

cap = cv2.VideoCapture('mycarplate.mp4')
 
area = [(27, 330), (16, 456), (1015, 451), (992, 330)]
count = 0


processed_numbers = set()
nbvoiture=0
mode=1
with open("car_plate_data.txt", "a") as file:
                        file.write("NbP NumberPlate  \tPrec_p\tMarque\t    Prec_m\t  Couleur\t   Date\t   Time\n")
list1 = []
"""
structure = myclass(1,"DL 7C D 5017",0.934,"inconnu",0.0,"Unknown","2024-05-02 23:53:15")
list1.append(structure)
structure = myclass(2,"DL 3C BJ 1384",0.927,"Suzuki",0.0,"Unknown","2024-05-02 23:53:32")
list1.append(structure)
structure = myclass(3,"DLZCATL762",0.883,"Nissan",0.0,"Unknown","2024-05-02 23:53:44")
list1.append(structure)
structure = myclass(4,"HR26C06869",0.574,"0Mitsubishi",0.0,"Unknown","2024-05-02 23:53:58")
list1.append(structure)
for item in list1:
   print(f"Numero: {item.numero}, Nb Plaque: {item.nomplaque},Prec_p :{item.precision},Marque: {item.marque},Prec_m :{item.marque_prcision},Couleur :{item.couleur} ,Date et Heure: {item.datetime}")
"""
while True: 
     
    ret, frame = cap.read()
    count += 1
    if count % 5 != 0:
        continue
    if not ret:
       break
    frame = cv2.resize(frame, (1020, 500))
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    for _, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        cx,cy = int(x1 + x2) // 2, int(y1 + y2) // 2
        result = cv2.pointPolygonTest(np.array(area, np.int32), ((cx, cy)), False)
        if result >= 0 :
            cv2.imshow("TEST", frame)
            crop = frame[y1:y2, x1:x2]
            gray1 = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
            reader = easyocr.Reader(['en'])
            results = reader.readtext(gray1)
            text = results[0][1]
            precision=f"{results[0][2]:.3f}"
            text = text.replace('(', '').replace(')', '').replace(',', '')
            if text not in processed_numbers:
                 
                if mode==1 and nbvoiture<NBM :
                    print("mode1 :",mode)  
                    print("S il vous plait deriger vers la place numero:",indice_value(places,False)+1)
                    nbvoiture+=1
                    processed_numbers.add(text) 
                    #couleur=color_detection1(frame)
                    couleur="None"
                    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    marque,prec=marque_detection(frame)
                    structure = myclass(indice_value(places,False)+1, text, precision,marque,prec,couleur,current_datetime)
                    
                    list1.append(structure)
                    with open("car_plate_data.txt", "a") as file:
                        
                        file.write(f"{str(indice_value(places,False)+1) + ':':<4}" +                         
                              f"{text.ljust(16)}" +                             
                              f"{str(precision)[:5].ljust(8)}" +                     
                              f"{marque.ljust(12)}" +                               
                              f"{str(prec)[:5].ljust(10)}" +                    
                              f"{couleur.ljust(14)}" +                       
                              f"{current_datetime}\n")
                    places[indice_value(places,False)]=True
                elif nbvoiture>=NBM :
                    print("Sorry le parking est sature...")
                elif mode ==0 :
                    sortie=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    entree=list1[indice_text(list1,text)].datetime
                    prix=int(prix_f(entree,sortie))
                    nbvoiture-=1
                    places[list1[indice_text(list1,text)].numero-1]=False
                    print("Veillez payer s il vous plait :",prix)
                    while 1:
                        d=int(input())
                        if d==prix :
                            break
                    efface("car_plate_data.txt", text)
                    for instance in list1:
                        if instance.nomplaque == text:
                            list1.remove(instance)
                    for item in list1:
                         print(f"Numero: {item.numero}, Nb Plaque: {item.nomplaque},Prec_p :{item.precision},Marque: {item.marque},Prec_m :{item.marque_prcision},Couleur :{item.couleur} ,Date et Heure: {item.datetime}")
    cv2.polylines(frame, [np.array(area, np.int32)], True, (6, 255, 255), 2)
    cv2.imshow("TEST", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
for item in list1:
    print(f"Numero: {item.numero}, Nb Plaque: {item.nomplaque},Prec_p :{item.precision},Marque: {item.marque},Prec_m :{item.marque_prcision},Couleur :{item.couleur} ,Date et Heure: {item.datetime}")
cap.release()    
cv2.destroyAllWindows()