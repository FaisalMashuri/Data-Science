# Laporan Proyek Machine Learning - Nama Anda

## Domain Proyek
Kegagalan untuk mengidentifikasi risiko kredit menyebabkan hilangnya pendapatan dan memperluas risiko kredit macet menjadi ancaman bagi profitabilitas. Kesalahan dalam analisis kredit menimbulkan risiko kredit, seperti kehilangan nasabah, ketidakpastian pengembalian pinjaman, bahkan ketidakmampuan nasabah untuk mengembalikan pinjaman (Zurada & Kunene, 2011).


Teknik klasifikasi pada machine learning dapat digunakan untuk menentukan risiko kredit. Misalnya di bank, sifat nasabah yang bisa membayar pinjamannya dapat di prediksi dan model dapat dibuat dengan menggunakan kumpulan data sebelumnya tentang pendanaan nasabah. Setelah itu model dapat digunakan pada yang baru. Pelanggan untuk menentukan kemungkinan membayar kembali kredit mereka. Dalam metode deskriptif, hubungan dapat dicari antara dua kumpulan data, misalnya kebiasaan berbelanja dari dua budaya yang berbeda dapat diselidiki kesamaannya.

Dalam memprediksi resiko pemberian kredit, maka diperlukannya sebuah sistem yang dapat memprediksi hal tersebut seperti machine learning. Teknik machine learning menjadi sangat populer saat ini karena ketersediaannya luas dalam kuantitas data yang besar serta kebutuhan kita untuk dapat mengubah data menjadi suatu pengetahuan. Kebutuhan akan pengetahuan inilah yang akhirnya mendorong industri TI untuk menggunakan machine learning. Data mining adalah proses mencari pola atau informasi. 

Terdapat beberapa penelitian terdahulu, seperti pada penelitian yang dilakukan oleh [Mittal et. al. (2016)](https://www.semanticscholar.org/paper/PREDICTION-OF-CREDIT-RISK-EVALUATION-USING-NAIVE-%2C-Mittal-Gupta/d92cc3a257e118b6eb27781d8c9db069a9433e9e), dimana dalam penelitian ini digunakan tiga metode yaitu SVM, Neural Network, dan juga Naive Bayes untuk menentukan algoritme yang paling optimal dan akurat utuk memprediksi risiko kredit guna membantu manajemen bank mengurangi kerugian dengan melakukan proses pengambilan keputusan yang tepat. Pada penelitian ini SVM diimplementasikan menggunakan polynomial kernel dengan parameter degree bernilai 3, gamma bernilai 2,6, 2,5, dan 3,2, serta nilai coef adalah 100, Neural Network diimplementasikan dengan iterasi sebanyak 500, nilai momentum constant sebesar 0,7, learning rate sebesar 0,1, dan hidden layer sebanyak 5, sedangkan Naive Bayes diimplementasikan dengan sekali proses. Hasil yang didapatkan adalah SVM menghasilkan tingkat akurasi tertinggi sebesar 92%, diikuti Naive Bayes sebesar 87%, dan Neural Network menghasilkan tingkat akurasi terendah sebesar 85%. Penelitian kedua merupakan penelitian yang dilakukan oleh Chakraborti (2014), dimana dalam penelitian ini menggunakan metode Naive Bayes, Decision Tree, SVM, Bayesian Network, dan Neural Network yang dibandingkan guna menemukan algoritme yang optimal dan akurat untuk memprediksi kelas gaji dari karyawan. 10-fold cross validation diimplementasikan pada semua algoritme yang digunakan pada penelitian ini, sedangkan khusus untuk SVM diimplementasikan menggunakan polykernel dengan nilai eksponen sebesar 1, chaceSize sebesar 250007, complexity constant sebesar 1, dan epsilon sebesar 1.0E-12, sedangkan untuk Neural Network diimplementasikan dengan nilai iterasi sebanyak 500 kali, learning rate sebesar 0,3, momentum constant sebesar 0,2, dan terdiri dari 54 node, satu hidden layer, satu input dan output layer. Hasil yang didapatkan, SVM menghasilkan tingkat akurasi tertinggi dengan 84,9022%, kemudian Naive Bayes menghasilkan 83,428%, dan Neural Network menghasilkan tingkat akurasi terendah dibanding SVM dan Naive Bayes sebesar 82,8936%.

Berdasarkan paparan diatas, terlihat SVM mendapatkan akurasi paling tinggi, oleh karena itu saya akan menggunkan algoritma itu dan membandingkannya dengan algoritma Logistic Regression untuk mendapatkan model terbaik dalam menyelesaikan masalah ini.


## Business Understanding

### Problem Statements
Kegagalan untuk mengidentifikasi risiko kredit menyebabkan hilangnya pendapatan dan memperluas risiko kredit macet menjadi ancaman bagi profitabilitas. Kesalahan dalam analisis kredit menimbulkan risiko kredit, seperti kehilangan nasabah, ketidakpastian pengembalian pinjaman, bahkan ketidakmampuan nasabah untuk mengembalikan pinjaman (Zurada & Kunene, 2011).


### Goals
Adapun tujuan yang ingin dicapai dari pembuatan laporan tugas akhir ini
adalah:
1. Mengimplementasikan metode SVM dan Logistic Regression untuk memprediksi
resiko jika ada kemungkinan kredit macet.
2. Melakukan proses untuk mendapatkan keluaran berupa status “Layak”
atau “Tidak Layak” calon penerima kredit dengan menggunakan SVM dan Logistic Regression

### Solution statements
- **Support Vector Machine ( SVM )**. salah satu metode dalam supervised learning yang biasanya digunakan untuk klasifikasi (seperti Support Vector Classification) dan regresi (Support Vector Regression). Dalam pemodelan klasifikasi, SVM memiliki konsep yang lebih matang dan lebih jelas secara matematis dibandingkan dengan teknik-teknik klasifikasi lainnya. SVM juga dapat mengatasi masalah klasifikasi dan regresi dengan linear maupun non linear.SVM digunakan untuk mencari hyperplane terbaik dengan memaksimalkan jarak antar kelas. Hyperplane adalah sebuah fungsi yang dapat digunakan untuk pemisah antar kelas. Dalam 2-D fungsi yang digunakan untuk klasifikasi antar kelas disebut sebagai line whereas, fungsi yang digunakan untuk klasifikasi antas kelas dalam 3-D disebut plane similarly, sedangan fungsi yang digunakan untuk klasifikasi di dalam ruang kelas dimensi yang lebih tinggi di sebut hyperplane. Hyperplane yang ditemukan SVM diilustrasikan seperti Gambar 1 posisinya berada ditengah-tengah antara dua kelas, artinya jarak antara hyperplane dengan objek-objek data berbeda dengan kelas yang berdekatan (terluar) yang diberi tanda bulat kosong dan positif. Dalam SVM objek data terluar yang paling dekat dengan hyperplane disebut support vector. Objek yang disebut support vector paling sulit diklasifikasikan dikarenakan posisi yang hampir tumpang tindih (overlap) dengan kelas lain. Mengingat sifatnya yang kritis, hanya support vector inilah yang diperhitungkan untuk menemukan hyperplane yang paling optimal oleh SVM.
- **Logistic Regression**. suatu cara permodelan masalah keterhubungan antara suatu variabel independen terhadap variabel dependen. Contohnya adalah menentukan apakah suatu nilai ukuran tumor tertentu termasuk kedalam tumor ganas atau tidak.suatu fungsi yang dibentuk dengan menyamakan nilai Y pada Linear Function dengan nilai Y pada Sigmoid Function. Tujuan dari Logistic Function adalah merepresentasikan data-data yang kita miliki kedalam bentuk fungsi Sigmoid. Kita dapat membentuk Logistic Function dengan melakukan langkah-langkah berikut:
    1. Melakukan opersai Invers pada Sigmoid Function, sehingga fungsi sigmoid berubah bentuk menjadi Y = ln(p/(1-p).
    2. Setarakan dengan fungsi Linear Y = b0+b1*X sehingga kita dapatkan persamaan ln(p/(1-p) = b0+b1*X.
    3. Ubah persamaan ln(p/(1-p) = b0+b1*X kedalam bentuk logaritmik sehingga didapatkan persamaan P = 1/(1+e^-(b0+b1*X)) 

## Data Understanding
Untuk data set saya menggunakan dataset dari kaggle yang dapat dilihat pada dibawah ini :
| Jenis | Keterangan | 
| ----------- | :---------: | 
| Sumber | https://www.kaggle.com/rikdifos/credit-card-approval-prediction | 
| Kategori | Layak (0) dan Tidak Layak(1)| 

Variabel-variabel pada dataset adalah sebagai berikut:
1. ID: Unique Id of the row
2. CODE_GENDER: Gender of the applicant. M is male and F is female.
3. FLAG_OWN_CAR: Is an applicant with a car. Y is Yes and N is NO.
4. FLAG_OWN_REALTY: Is an applicant with realty. Y is Yes and N is No.
5. CNT_CHILDREN: Count of children.
6. AMT_INCOME_TOTAL: the amount of the income.
7. NAME_INCOME_TYPE: The type of income (5 types in total).
8. NAME_EDUCATION_TYPE: The type of education (5 types in total).
9. NAME_FAMILY_STATUS: The type of family status (6 types in total).
10. DAYS_BIRTH: The number of the days from birth (Negative values).
11. DAYS_EMPLOYED: The number of the days from employed (Negative values). This column has error values.
12. FLAG_MOBIL: Is an applicant with a mobile. 1 is True and 0 is False.
13. FLAG_WORK_PHONE: Is an applicant with a work phone. 1 is True and 0 is False.
14. FLAG_PHONE: Is an applicant with a phone. 1 is True and 0 is False.
15. FLAG_EMAIL: Is an applicant with a email. 1 is True and 0 is False.
16. OCCUPATION_TYPE: The type of occupation (19 types in total). This column has missing values.
17. CNT_FAM_MEMBERS: The count of family members.

### Visualisasi
![output_70_0](https://user-images.githubusercontent.com/62064078/139354726-41e086a4-fc87-4013-b21c-84929db94007.png)
pada diagram diatas menunjukan kalau kebanayakan yang mengajukan credit card adalah yang bergender perempuan

## Data Preparation
Pada tahap data preparation saya menggunakan beberapa teknik yaitu :
- Handling missing value: hal ini dilakukan karena terdapat banyak sekali data yang kosong, untuk itu pada proses ini record akan dihapus jika memiliki data yang kosong pada tiap kolomnya
- Data Cleaning : proses mengidentifikasi dan mengoreksi data yang rusak, tidak lengkap, duplikat, salah, dan tidak relevan. Sedangkan yang kamu lakukan merupakan proses seleksi fitur. Seleksi fitur adalah teknik untuk memilih fitur / kolom penting dan relevan terhadap data dan mengurangi fitur yang tidak relevan .

    Kolom yang akan kita hapus adalah :
- occupation type : kolom ini dihapus karena banyak sekari terdapat missing value, jadi bisa dibilang kolom ini tidak memiliki peran penting terhadap prediksi yang akan dilakukan
- Flag_mobil : kolom ini tidak berkorelasi dengan problem yang akan kita selesaikan karena diterima atau tidaknya pengajuan kredit seseorang tidak bergantung pada apakah dia punya miobil atau tidak, misal dia punya mobil tapi mobil yang harganya sudah dibawah jumlah kredit yang ajukan maka itu tidak bisa jadi jaminan , untuk itu Kolom ini dihapus 
- Flag_work_phone : tidak memiliki korelasi dengan problem yang akan diselesaikan dan juga Kolom ini hanya berisi nilai 0 & 1 untuk Seluler yang tidak dikirimkan, oleh karena itu hapus kolom
- flag_phone : tidak memiliki korelasi dengan problem yang akan diselesaikan karena diterima atau tidak kredit seseorang tidak bergantung apakah dia punya handphone atau tidak, oleh karena itu hapus kolom
- flag_email : tidak memiliki korelasi dengan problem yang akan diselesaikan karena diterima atau tidak kredit seseorang tidak bergantung apakah dia punya alamat email atau tidak, oleh karena itu hapus kolom.
- Handling Outliers : pada tahap ini saya akan menghapus outliers yang terdapat pada beberapa kolom,karena outliers akan mengkibatkan bias pada hasil prediksi yang akan kita lakukan maka kolom ini akan kita hapus. Kolom yang terdapat outliers diantara lain :
    AMT_INCOME_TOTAL : 5276
    CNT_CHILDREN: 6075
    YEARS_EMPLOYED : 9531
    CNT_FAM_MEMBERS : 5690
    
- Menggunakan metode SMOTE untuk membuat data menjadi balanced
- Lalu menormalisasikan data dengan menggunakan MinMaxScaler agar datanya menjadi pada rentang yang sama yaitu 0 sampai 1
- Setelah itu mebagi data menjadi data train dan juga data test

## Modeling
1. untuk model pertama menggunakna algoritma Logistic Regression dengan parameter solver="liblinear" dan random_state=4. Setelah dilatih dengan data training model menghasilkan accuracy sebesar 62,3%.
2. Model kedua menggunakan algoritma SVM dengan parameter kernel="linear" mendapatkan accuracy sebesar 63%.

Lalu dilakukan hyperparameter tunning pada kedua model
1. untuk Logistic Regression mednapatkan akurasi sebesar 64%
2. SVM mendapatkan accuracy sebesar 70%

## Evaluation
setelah ditraining lalu model di evaluasi kedalam beberapa metrik yaitu
1. Logistic Regression :
    accuracy : 64%
    recall: 64%
    precision : 63%
    f1-score : 63%
 
2. SVM :
    accuracy : 70%
    precision: 67%
    recall : 79%
    f1-score : 72%

### Metriks Evaluasi pada Binary Classification
- Confusion matrix
- Accuracy
- Precission & Recall
- F1 Score

**Confusion Matrix**
![conf_metrics](https://user-images.githubusercontent.com/62064078/139354746-777fd8d8-c4b8-4091-b086-323c1c38a42b.png)
- True Positive (TP): data yang diklasifikasikan dengan tepat sebagai keluaran positif dan benar
- False Negatif (FN) adalah data yang diklasifikasikan dengan kurang tepat
- False Positif (FP) adalah data yang diklasifikasikan kurang tepat apabila keluaran berupa positif atau benar
- True Negatif (TN) adalah data yang diklasifikasikan dengan tepat sebagaimana keluaran negatif atau salah.


**Accuracy**
Penjelasan mengenai metric accuracy telah disinggung sebelumnya pada awal pembahan classification metric. Secara gampang, accuracy adalah seberapa akurat model mengklasifikasikan dengan benar. Secara matematis metric ini dapat diperoleh dengan rumus di bawah ini.
![accuracy](https://user-images.githubusercontent.com/62064078/139354669-a6cb401f-fadc-4dd1-91d7-db922a2aefe9.PNG)


**Precision**<br>
Precision adalah suatu metric yang menggambarkan akurasi data yang diminta dengan hasil prediksi yang diberikan oleh model. Secara matematis nilai precision dapat diperoleh dengan menggunakan cara di bawah ini.
![precision](https://user-images.githubusercontent.com/62064078/139354679-cb166ff5-5150-44db-b67a-40571e7b45ca.PNG)


**Recal or True Positive Rate(TPR) or Sensivity**<br>
Recall adalah suatu metric yang menggambarkan akurasi hasil prediksi dari model yang dibandingkan dengan keseluruhan jumlah ground truth dari data kelas tersebut. Nilai recall dapat diperoleh dengan cara di bawah ini.
![recal](https://user-images.githubusercontent.com/62064078/139354684-c2050727-2f32-41ee-a97d-a49d83a014d8.PNG)

**F1-Score**<br>
F-1 Score atau bisa disebut juga dengan F-1 Measurement adalah metrics yang menggambarkan perbandingan rata-rata precision dan recall yang harmonik. Metrics ini penting untuk mengetahui keharmonisan dari kedua metric tersebut. Hal ini karena jika nilai precision besar tetapi nilai recall kecil, prediksi dari model akurat tetapi tidak mencakup seluruh kemungkinan ground truth yang ada. Di lain sisi, apabila nilai recall tinggi tetapi tidak diikuti dengan nilai precision yang baik, prediksi dari model mencakup seluruh kemungkinan ground truth tetapi tidak diikuti dengan prediksi yang benar. Oleh karena itu, metric F-1 Score dapat menjadi faktor penentu keseimbangan antara kedua metric tersebut. Nilai F-1 Score dapat diperoleh dengan cara seperti di bawah ini.
![f1-score](https://user-images.githubusercontent.com/62064078/139354695-74303376-252d-47c8-8281-72b9b07371d7.PNG)



## Kesimpulan
untuk percobaan kali ini model SVM merupakan model yang paling cocok untuk diterapkan pada masalah ini karena dapat dilihat dari evaluasi model SVM unggul jauh dari Logistic Regression dari segala metric ( accuracy, precision, recall, f1-score )

**---Ini adalah bagian akhir laporan---**




```python

```
