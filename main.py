import csv # CSV dosyalarını okumak için gerekli olan kütüphane
import numpy as np # Numpy kütüphanesi, dizi ve matris işlemleri için kullanılır
from matplotlib import pyplot as plt # Matplotlib kütüphanesi, grafik çizme işlemleri için kullanılır
import math # Matematiksel işlemler için kullanılan kütüphane
from collections import defaultdict # Varsayılan değerleri ayarlamaya yarayan kütüphane
import random # Rastgele sayı üretmek için kullanılan kütüphane
import tkinter as tk # Dosya açma penceresini açmak için kullanılan kütüphane
from tkinter import filedialog # Dosya açma penceresini açmak için kullanılan kütüphane
import traceback # Hata ayıklama işlemleri için kullanılan kütüphane
import os # Sistem özelliklerine ulaşmak için kullanılan kütüphane

class Clusterer: # Clusterer sınıfı oluşturuluyor
    def __init__(self, filename): # init metodu ile sınıfın ilk çalıştırılışında yapılacak işlemler tanımlanıyor
        self.data = self.read_data(filename) 

    def read_data(self, filename):     # CSV dosyasındaki verinin okunmasını sağlayan metod
        with open(filename, 'r') as f: 
            reader = csv.reader(f)     # CSV okuyucu nesnesi oluşturuluyor
            headers = next(reader)    # CSV dosyasındaki ilk satır, yani sütun başlıkları okunuyor
            rows = [row for row in reader]  # CSV dosyasındaki geri kalan satırlar okunuyor
        return headers, rows           

    def select_columns(self): # Kullanıcıpip install pipreqsan hangi sütunların kullanılacağının sorulmasını sağlayan metod
        headers, _ = self.data 
        print("Seçilebilir sütunlar:") 
        for i, header in enumerate(headers): 
            print(f"{i}: {header}") 
        column_indices = input("x ekseni ve y ekseni için seçilecek sıralı ikiliyi giriniz. (Girdiğiniz komut virgül ile ayrılmış olmalıdır. Örneğin: 0,3): ")
        
        column_indices = [int(i) for i in column_indices.split(',')] # Girilen değerler virgülle ayrılıp int tipine çeviriliyor
        self.selected_columns = [headers[i] for i in column_indices] # Seçilen sütunların isimleri self.selected_columns değişkenine atanıyor
        return column_indices # Seçilen sütun numaraları döndürülüyor

    def preprocess(self,column_indices = None):   # Verinin ön işleme yapılmasını sağlayan metod
        _, rows = self.data                     
        if(column_indices == None):              # Eğer sütun numaraları belirtilmemişse
            X = [[float(row[i]) for i in range(len(row))] for row in rows]   # Tüm sütunlar için veriler float tipine çevirilerek 2D diziye atanıyor
            return np.array(X)                   
        else:                                   # Eğer sütun numaraları belirtilmişse
            X = [[float(row[i]) for i in column_indices] for row in rows]   # Belirtilen sütunlar için veriler float tipine çevirilerek 2D diziye atanıyor
            return np.array(X)                  

    def cluster(self, X):
        # Küme sayısının belirlenmesi
        k = self.determine_k()
        
        # İlk küme merkezini rastgele seçilmesi
        cluster_centers = [X[np.random.randint(len(X))]]

        
        # Kalan küme merkezlerinin k-means yöntemiyle seçilmesi
        for i in range(1, k):
            # Her nokta ile en yakın küme merkezinin mesafelerinin hesaplanması
            distances = np.array([np.min([np.linalg.norm(x - center) for center in cluster_centers]) for x in X])
            # Kareleri toplamına oranına göre sonraki küme merkezlerinin seçilmesi
            cluster_centers.append(X[np.random.choice(range(len(X)), p=distances**2/np.sum(distances**2))])

        # k-means kümeleme işleminin yapılması
        while True:
            # Her noktayı en yakın küme merkezine ataması
            cluster_labels = []
            for x in X:
                distances = [np.linalg.norm(x - cluster_center) for cluster_center in cluster_centers]
                cluster_label = np.argmin(distances)
                cluster_labels.append(cluster_label)

            # Yeni küme merkezlerinin hesaplanması
            new_cluster_centers = np.zeros((k, len(X[0])))
            for i in range(k):
                points = [X[j] for j in range(len(X)) if cluster_labels[j] == i]
                if len(points) > 0:
                    new_cluster_centers[i] = np.mean(points, axis=0)

            # Kümelemenin yakınsayıp birleşmediğinin kontrolü
            if np.all(new_cluster_centers == cluster_centers):
                break
            else:
                cluster_centers = new_cluster_centers

        return cluster_labels, cluster_centers

    def normalize_columns(self, X):
        # Normalize each column by applying the formula (x - x.min()) / (x.max() - x.min())
        X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        return X

    def determine_k(self):
        k = int(input("Kaç tane küme olacağını giriniz: "))
        return k


    def plot_scatter(self, X=None, cluster_labels=None, cluster_centers=None, selected_columns = None):
      # Verilerin çiftleri olarak çıkarılması
      headers, rows = self.data
      if(selected_columns == None):
        # Grafikte kullanılacak sütunların indekslerinin seçilmesi
        column_indices = self.select_columns()
        if cluster_centers is None:
            print("Algoritmayı çalıştırmadığınızdan dolayı sadece seçilen x ve y'ye göre tabloda değerler gösterilecektir.")
            # Grafiğe ait verilerin çıkarılması
            data = [[float(row[i]) for i in column_indices] for row in rows]

            # Figure ve axis oluşturulması
            fig, ax = plt.subplots()

            # Verilerin çizdirilmesi
            ax.scatter(*zip(*data))

            plt.show()
        else:
            k = len(cluster_centers)
            # Grafiğe ait verilerin çıkarılması
            data = [[float(X[j][i]) for i in column_indices] for j in range(len(X))]

            # Her küme için renklerin belirlenmesi
            # HTML renk kodlarından oluşan bir liste oluşturulması
            colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k'] + [f'#{random.randint(0, 0xFFFFFF):06x}' for _ in range(k - 7)]


            # Figure ve axis oluşturulması
            fig, ax = plt.subplots()

            # Her kümedeki noktaların çizdirilmesi
            for i in range(k):
                points = np.array([data[j] for j in range(len(data)) if cluster_labels[j] == i])
                ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i], label=f"Küme {i+1}")
            # Küme merkezlerinin çizdirilmesi
            ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='*', s=200, c='#050505')
            # Grafiğe açıklama eklenmesi
            ax.legend(loc='upper left')
            print("Yıldız ile gösterilenler tüm sütunlar dikkate alınarak oluşturulmuş kümelerin Geometrik Merkezidir(Centroid).")

            plt.show()
      else:
            k = len(cluster_centers)
            # Grafiğe ait verilerin çıkarılması
            data = X
            # Her küme için renklerin belirlenmesi
            # HTML renk kodlarından oluşan bir liste oluşturulması
            colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k'] + [f'#{random.randint(0, 0xFFFFFF):06x}' for _ in range(k - 7)]

            # Figure ve axis oluşturulması
            fig, ax = plt.subplots()

            # Her kümedeki noktaların çizdirilmesi
            for i in range(k):
                points = np.array([data[j] for j in range(len(data)) if cluster_labels[j] == i])
                ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i], label=f"Küme {i+1}")
            # Küme merkezlerinin çizdirilmesi
            ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='*', s=200, c='#050505')
            # Grafiğe açıklama eklenmesi
            ax.legend(loc='upper left')
            print("Yıldız ile gösterilenler seçilen sütunlar dikkate alınarak oluşturulmuş kümelerin Geometrik Merkezidir(Centroid).")

            plt.show()

    def calculate_wcss(self, cluster_centers, clusters, X): # WCSS (Within Cluster Sum of Squares) değerlerinin hesaplanmasını sağlayan metod
        wcss = 0 

        k = len(cluster_centers)  

        # Her küme için döngüyü çalıştır
        for i in range(k):
            # Her küme için X dizisinden küme etiketi i olan verileri al
            points = [X[j] for j in range(len(X)) if clusters[j] == i]
            points = np.array(points)  
            # Küme merkezlerinden verilere olan uzaklıkların karesi alınıp WCSS değerine toplanıyor
            wcss += np.sum((points - cluster_centers[i])**2)
        return wcss  

    def calculate_bcss(self, clusters, X): # BCSS (Between Cluster Sum of Squares) değerlerinin hesaplanmasını sağlayan metod   
        bcss = 0 
        
        k = len(np.unique(clusters))

        # Tüm verilerin ortalamasını al
        global_mean = np.mean(X, axis=0)

        # Her küme için döngüyü çalıştır
        for i in range(k):
            # Her küme için X dizisinden küme etiketi i olan verileri al
            points = [X[j] for j in range(len(X)) if clusters[j] == i]
            points = np.array(points) 
            # Küme verilerinin ortalamasını al
            cluster_mean = np.mean(points, axis=0)
            # Küme ortalaması ve tüm verilerin ortalaması arasındaki uzaklığın karesi alınarak BCSS değerine toplanıyor
            bcss += len(points) * np.sum((cluster_mean - global_mean)**2)
        return bcss  
        
    def euclidean_distance(self, point_1, point_2): # İki nokta arasındaki euclid uzaklığını hesaplayan metod
        # Uzaklığı 0 olarak başlat
        dist = 0
        # İki noktanın koordinatlarını döngüyle dolaş
        for x in range(len(point_1)):
            # Koordinat farklarının karesini uzaklığa topla
            dist += (point_1[x] - point_2[x])**2
        # Uzaklığın karekökünü döndür
        return math.sqrt(dist)
    
    def calculate_dunn_index(self, cluster_centers, cluster_labels, data):  # Dunn endeksinin hesaplanmasını sağlayan metod
        diam_lst = []  # Kümelerin çapını tutan liste
        dist_lst = []  # Farklı kümelerdeki noktalar arasındaki en küçük uzaklığı tutan liste

        
        if len(dist_lst) == 0:
            min_dist = 1
        else:
            min_dist = min(dist_lst)

        # Kümelerin çaplarını hesapla
        for k in range(len(cluster_centers)):
            clust_diam_lst = []  # Küme içindeki veriler arasındaki en büyük uzaklığı tutan liste
            for x in range(len(data)):
                for y in range(len(data)):
                    # Eğer x ve y noktaları aynı kümeye aitse ve küme etiketleri k değerine eşitse
                    if cluster_labels[x] == cluster_labels[y] and cluster_labels[x] == k:
                        # x ve y noktaları arasındaki uzaklığı clust_diam_lst'ye ekle
                        clust_diam_lst.append(self.euclidean_distance(data[x], data[y]))
                        diam_lst.append(max(clust_diam_lst))  # clust_diam_lst'nin en büyük elemanını diam_lst'ye ekle
        max_diam = max(diam_lst)  # diam_lst'nin en büyük elemanını max_diam değişkenine ata

        # Farklı kümelerdeki noktalar arasındaki en küçük uzaklığı hesapla
        for i in range(len(cluster_centers)):
            for j in range(i+1, len(cluster_centers)):
                clust_dist_lst = []  # İki küme arasındaki verilerin uzaklıklarını tutan liste
                for x in range(len(data)):
                    for y in range(len(data)):
                        # Eğer x noktası i. kümeye ve y noktası j. kümeye aitse
                        if cluster_labels[x] == i and cluster_labels[y] == j:
                            clust_dist_lst.append(self.euclidean_distance(data[x], data[y]))
                dist_lst.append(min(clust_dist_lst))  # clust_dist_lst'nin en küçük elemanını dist_lst'ye ekle

        dunn_index = min_dist / max_diam  

        return dunn_index  

    def write_to_file(self, cluster_labels, wcss, bcss, dunn_index):  # Küme etiketlerini dosyaya yazdıran metod
        
        with open('Sonuc.txt', 'w') as f:
            for i, label in enumerate(cluster_labels):
                f.write(f"Kayıt {i+1}: Küme {label+1}\n")
            # Her bir kümede kaç kayıt olduğunu saymak için defaultdict oluştur
            cluster_counts = defaultdict(int)

            # Küme etiketlerini döngüyle gez
            for label in cluster_labels:
                # Küme sayısını arttır
                cluster_counts[label] += 1

            # Küme sayılarını sırala
            sorted_cluster_counts = sorted(cluster_counts.items())
            # Her bir küme için kayıt sayısını yazdır
            for label, count in sorted_cluster_counts:
                f.write(f"\nKüme {label+1}: {count} kayıt")
            f.write(f"\n\nWCSS: {wcss}\nBCSS: {bcss}\nDunn Index: {dunn_index}")
        # Çalışma dizinini al
        cwd = os.getcwd()

        print(f'Dosya şuraya kaydedildi: {cwd}') 

def main():
    # Veriyi oku
    root = tk.Tk()
    # Tkinter kök penceresini gizle
    root.withdraw()
    print("Final-data dosyasını seçiniz.")
    # Kullanıcıdan bir dosya seçmesi için iste
    file_path = filedialog.askopenfilename()
    
    clusterer = Clusterer(file_path)
    
    while True:
        
        choice = input("1) K-Means Algoritmasını Çalıştır.\n2) Kümeleri Görselleştir.\n3) Seçilen sütunlara göre k-means algoritmasını Çalıştır.\n4) Kümeleri temizle.\n5) Programdan çık.\nProgramın çalışması için menüdeki sayılardan birini seçiniz.\n")

        # k-means'i çalıştır
        if choice == '1':
            try:
                cluster_labels = None
                selected_columns = None
                # Veriyi ön işle
                X = clusterer.preprocess()
                X = clusterer.normalize_columns(X)

                # Kümelemeyi yap
                cluster_labels, cluster_centers = clusterer.cluster(X)
                
                # WCSS'yi hesapla
                wcss = clusterer.calculate_wcss(cluster_centers, cluster_labels,X)
                # BCSS'yi hesapla
                bcss = clusterer.calculate_bcss(cluster_labels, X)
                # Dunn İndex'ini hesapla
                dunn_index = clusterer.calculate_dunn_index(cluster_centers, cluster_labels,X)

                clusterer.write_to_file(cluster_labels,wcss,bcss,dunn_index)
                print("Cevaplar başarıyla dosyaya yazılmıştır ve artık kümelenmiş verileri görselleştirebilirsiniz.")
            except Exception as e:
                print(f"Program hata ile karşılaştı:")
                print(f"Hata: {type(e).__name__}")
                print(f"Mesaj: {e}")
                print(f"Line: {traceback.format_exc()}")

        # Kümeleri görselleştir
        elif choice == '2':
            try:
                if (cluster_labels) is None:
                    clusterer.plot_scatter()
                else:

                    clusterer.plot_scatter(X, cluster_labels, cluster_centers,selected_columns)
                    
            except Exception as e:
                print(f"Hata: {type(e).__name__}")
                print(f"Mesaj: {e}")
                print(f"Line: {traceback.format_exc()}")
        
        elif choice == '3':
          cluster_labels = None
          selected_columns = None
          # Kümeleme için kolonları seç
          column_indices = clusterer.select_columns()
          selected_columns = column_indices
          # Veriyi ön işle
          X = clusterer.preprocess(column_indices)
          X = clusterer.normalize_columns(X)

          # Kümelemeyi yap
          cluster_labels, cluster_centers = clusterer.cluster(X)
          
          # WCSS'yi hesapla
          wcss = clusterer.calculate_wcss(cluster_centers, cluster_labels,X)
          # BCSS'yi hesapla
          bcss = clusterer.calculate_bcss(cluster_labels, X)
          # Dunn İndex'ini hesapla
          dunn_index = clusterer.calculate_dunn_index(cluster_centers, cluster_labels,X)

          clusterer.write_to_file(cluster_labels,wcss,bcss,dunn_index)
          print("Cevaplar başarıyla dosyaya yazılmıştır ve artık kümelenmiş verileri görselleştirebilirsiniz.")
        
        # Kümeleri temizle
        elif choice == '4':
            cluster_labels = None
            selected_columns = None
            print("Kümeler temizlendi.")
        
        # Programdan çık
        elif choice == '5':
            print("Programdan çıkılıyor...")
            break
        else:
            print("Lütfen geçerli bir seçenek giriniz.")

if __name__ == '__main__':
    main()