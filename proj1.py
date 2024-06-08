import cv2
import face_recognition

img = cv2.imread("./data_img/japa.jpeg")
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
encodar = face_recognition.face_encodings(rgb_img)[0]

img2 = cv2.imread("./data_img/rodney_barbosa2.jpeg")
rgb_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
encodar2 = face_recognition.face_encodings(rgb_img2)[0]

resultado = face_recognition.compare_faces([encodar], encodar2)
print("Resultado: ", resultado)

cv2.imshow("rodney", rgb_img2)
cv2.waitKey(0)
