class TakipEdilenAracClass:
    def __init__(self, aracID, kutle_merkezi,sinif):
        self.aracID = aracID
        self.kutle_merkezleri = [kutle_merkezi]
        self.sinif  = sinif
        self.aracSayildi = False
        self.aracSayildiRoi = -1
        self.ilkYonSifirlandi = False #ikinci tespitte karar verilmesin
        self.yDizisi = []
        self.ilkGecis = -1          # giris-cikis analizleri icin
        self.ikinciGecis = -1