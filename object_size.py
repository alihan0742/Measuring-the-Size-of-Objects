# Paketleri import etme
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# Argümanı ayrıştırma ve argümanları çözümleme işlemi
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
ap.add_argument("-w", "--width", type=float, required=True,
	help="width of the left-most object in the image (in inches)")
args = vars(ap.parse_args())

# Görüntünün yüklenmesi, gri tonlamaya dönüştürme ve bulanıklaştırma
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 0)

# Kenar tespiti
# Nesne kenarları arasındaki boşlukları kapat
edged = cv2.Canny(gray, 50, 100)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)

# Kenar haritasında kontürleri bul
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# Kontürleri soldan sağa doğru sıralayın ve
# 'metrik başına piksel' kalibrasyon değişkeni ayarı
(cnts, _) = contours.sort_contours(cnts)
pixelsPerMetric = None

# Ayrı ayrı konturlar üzerinde döngü
for c in cnts:
	# Kontur yeterince büyük değilse, yoksay
	if cv2.contourArea(c) < 100:
		continue

	# Konturun döndürülmüş sınırlayıcı kutusunu hesapla
	orig = image.copy()
	box = cv2.minAreaRect(c)
	box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
	box = np.array(box, dtype="int")

	# konturdaki noktaları görünecek şekilde sıralama
	# sol üst, sağ üst, sağ alt ve sol alt
	# Sonra döndürülen sınırlamanın anahatlarını çiz

	box = perspective.order_points(box)
	cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

	# Orijinal noktaların üzerinden geçerek bunları çiz
	for (x, y) in box:
		cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

	# Sınırlama kutusunu açtıktan sonra orta noktayı hesaplayın
	# üst sol ve üst sağ koordinatlar arasında, ardından
	# sol alt ve sağ alt koordinatlar arasındaki orta nokta hesaplanır
	(tl, tr, br, bl) = box
	(tltrX, tltrY) = midpoint(tl, tr)
	(blbrX, blbrY) = midpoint(bl, br)

	# Sol üst ve sağ üst noktalar arasındaki orta noktayı hesaplamak,
	# sonra sağ üst ve sağ alt arasındaki orta nokta
	(tlblX, tlblY) = midpoint(tl, bl)
	(trbrX, trbrY) = midpoint(tr, br)

	# Resmin üzerine orta noktaları çiz
	cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

	# Orta noktaların arasına çizgi çiz
	cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
		(255, 0, 255), 2)
	cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
		(255, 0, 255), 2)

	#Orta noktalar arasındaki Öklid mesafesini hesaplar
	dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
	dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

	# Metrik başına piksel başlatılmadıysa, o zaman
	# piksellerin sağlanan metriğe oranı olarak hesapla
	# (bu durumda, inç)
	if pixelsPerMetric is None:
		pixelsPerMetric = dB / args["width"]

	# nesnenin boyutunu hesapla
	dimA = dA / pixelsPerMetric
	dimB = dB / pixelsPerMetric

	# Resimdeki nesne boyutlarını çiz
	cv2.putText(orig, "{:.1f}in".format(dimA),
		(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
		0.65, (255, 255, 255), 2)
	cv2.putText(orig, "{:.1f}in".format(dimB),
		(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
		0.65, (255, 255, 255), 2)

	# çıktı görüntüsü
	cv2.imshow("Image", orig)
	cv2.waitKey(0)
