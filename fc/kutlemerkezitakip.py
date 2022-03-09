from abc import abstractclassmethod
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

class KutleMerkeziTakipClass:
	def __init__(self, maksAracinGorunmedigiFrameSayisi, maksUzaklik):
		
        
        
        #aracın ID'si tanımlanır. araç ID'lerini araç kutle merkezine 
        #eşlemek için kullanılan sözlük oluşturulur. Ardından,
        #aracınn kac adet ardışık frame'de kaybolduğunu tutan sözlük
        #oluşturulur
        #Son olarak, arac ID'lerini ve bu ID'ye karşılık gelen araçların sınıfları oluşturulur.
		
		self.siradakiAracID = 0
		self.araclar = OrderedDict()
		self.kayboldu = OrderedDict()
		self.aracSinif = OrderedDict()
        
	
        
        #araç silinene kadar kaç adet ardışık frame geçmesi
        #gerektiğini tutan sayı, bu değişkende tutulur
		self.maksAracinGorunmedigiFrameSayisi = maksAracinGorunmedigiFrameSayisi

		
        #araci kendisiyle ilişkilendirmek için kütle merkezleri arasındaki maksimum mesafeyi tutan değişken.
        #eğer mesafe maksUzaklik'dan büyükse araç "kayboldu" olarak
        #işaretlenir
		self.maksUzaklik = maksUzaklik

	def arac_kaydet(self, centroid,aracSinif):
        #bir araç kaydedilirken, sıradaki uygun ID değeri kütle merkezini tutmak için seçilir
		self.araclar[self.siradakiAracID] = centroid
		self.aracSinif[self.siradakiAracID] = aracSinif 
		self.kayboldu[self.siradakiAracID] = 0
		self.siradakiAracID += 1

	def kayitSil(self, aracID):
        # bir araç ID'sinin kaydını silmek için iki sözkükten de silinir
		
		del self.araclar[aracID]
		del self.kayboldu[aracID]


	def guncelle(self, dikdortgenler):
		# giriş sınırlayıcı kutu dikdörtgenleri listesinin boş olup olmadığını kontrol et
		if len(dikdortgenler) == 0:
			# o anda takip edilmekte olan araç(lar) varsa kayboldu olarak işaretlenir.
			for aracID in list(self.kayboldu.keys()):
				self.kayboldu[aracID] += 1

				#maksimum sayıda arka arkaya belirli bir araç kayboldy olarak işaretlenmişse araç silinir
				if self.kayboldu[aracID] > self.maksAracinGorunmedigiFrameSayisi:
					self.kayitSil(aracID)
					

			#guncellenecek bir şey olmadığı için fonksiyondan dönülür.
			return self.araclar,self.aracSinif

		# görüntü için bir giriş dizisi oluşturulur
        #bu liste takip edilen araçların kütle merkezleri ile doldurulur
		girisListesi = []
		for dikdortgen in dikdortgenler:
			girisKutleMerkezleri = np.zeros(2, dtype="int")
			(baslangicX,baslangicY,bitisX,bitisY) = dikdortgen[0]
			merkezX = int((baslangicX + bitisX) / 2.0)
			merkezY = int((baslangicY + bitisY) / 2.0)
			girisKutleMerkezleri = (merkezX, merkezY)
                                                        
                                                      #dikdortgen[1] = class  
			girisListesi.append([girisKutleMerkezleri,dikdortgen[1]])

        #Şu anda herhangi bir araç takip edilmiyosa, giriş kütle merkezlerinin her biri kaydedilir.
		if len(self.araclar) == 0:
			
			for i in range(0, len(girisListesi)):
				self.arac_kaydet(girisListesi[i][0],girisListesi[i][1])

        # takip edilen araç(lar) varsa, var olan kütle merkezleriyle giriş kütle merkezleri eşleştirilir.
        
		else:
			# araç ID'lerini ve bu ID'lere karşılık gelen kütle merkezlerini al
			
			aracIDleri = list(self.araclar.keys())
			aracKutleMerkezleri = list(self.araclar.values())
			
			girisKutleMerkezi = np.zeros((len(girisListesi), 2), dtype="int")

			for i,x in enumerate(girisListesi):
				girisKutleMerkezi[i] = x[0]
                
            #her araç kütle merkezi ve giriş kütle merkezi çifti için, uzaklık hesaplanır
            #burada amac, bir giriş kütle merkezini var olan bir kütle merkeziyle eşleştirmektir
		
			
			
			
			uzaklik = dist.cdist(np.array(aracKutleMerkezleri), girisKutleMerkezi)

            #bu eşleştirmeyi yapabilmek için, ilk olarak dikeyde (satir) en düşük değeri bulmalıyız.
            #sonrasında, en düşük uzaklığa sahip satır değeri listenin en önüne gelecek şekilde değerleri sıralamalıyız
			
			satirlar = uzaklik.min(axis=1).argsort()
            
            #benzer bir işlemi sütunlar (yatayda) da yapmamız gerekir. değerleri sıralarız. ve en düşük mesaafeyi (kütle merkezine en yakın olan sütun değeri)
            #listenin başına alırız
			
			sutunlar = uzaklik.argmin(axis=1)[satirlar]
			
            
			#bir aracı güncellememize, kaydetmemize veya kaydını silmemize gerek olup olmadığını belirlemek için, halihazırda incelemiş olduğumuz satırları ve sütun dizinlerini dikkate almamız gerekir.
			#bu amaçla set (küme) tanımlanır. çünkü kümelerde 1 eleman 1'den fazla blunamaz
			kullanilanSatirlar = set()
			kullanilanSutunlar = set()

			# satir-sütun değerleri incelenir sıkıştırılır
			
			for (satir, sutun) in zip(satirlar, sutunlar):
			
                #eğer bu satır ya da sütun değerini daha önce kullanılmışsa görmezden gelinir
				
				if satir in kullanilanSatirlar or sutun in kullanilanSutunlar:
						
					continue

				
                
                #eğer iki kütle merkezi arasındaki uzaklık belirlenen maksimum mesafe değerinden büyükse, kütle merkezleri ilişkilendirilmez
				if uzaklik[satir, sutun] > self.maksUzaklik:
					
					continue

				#aksi durumda, o satır için araç ID'si alınır. yeni kütle merkezi belirlenir. kaybolma sayacı sıfırlanır 
              
				aracID = aracIDleri[satir]
				

				self.araclar[aracID] = girisListesi[sutun][0]

				self.kayboldu[aracID] = 0



				# bu satır-sütun da kullanıldığı için kullanılanlar listesine eklenir.
				kullanilanSatirlar.add(satir)
				kullanilanSutunlar.add(sutun)

			# henüz kullanılmayan satır ve sütun dizinlerini hesapla
			kullanilmayanSatirlar = set(range(0, uzaklik.shape[0])).difference(kullanilanSatirlar)
			kullanilmayanSutunlar = set(range(0, uzaklik.shape[1])).difference(kullanilanSutunlar)

            
            #araç kütle merkezi sayısı, giriş kütle merkezi sayısından büyük eşit ise, bazı araçlların kayıp olup olmadığı kontrol edilir
			
			if uzaklik.shape[0] >= uzaklik.shape[1]:
				
				# kullanılmayan satır indeksleri incelenir
				for satir in kullanilmayanSatirlar:
                
					#ilgili satır dizini için araç IDsi alınır ve kaybolma sayacı artırılır.
					aracID = aracIDleri[satir]
					self.kayboldu[aracID] += 1

					#aracın kaç frame boyunca arka arkaya görünmediği kontrol edilir. eğer limitin üstündeyse silinir.
					if self.kayboldu[aracID] > self.maksAracinGorunmedigiFrameSayisi:
						self.kayitSil(aracID)
						
            #aksi halde, her giriş kütle merkezini, takip edilen araç olarak kaydetmeliyiz
			
			else:
				for sutun in kullanilmayanSutunlar:
					
					self.arac_kaydet(girisListesi[sutun][0],girisListesi[sutun][1])

		# takip edilen araçlari döndür.
		return self.araclar,self.aracSinif