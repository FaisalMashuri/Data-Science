#!/usr/bin/env python
# coding: utf-8

# # Credit Card Approval Prediction
# 
# Kredit bagi masyarakat bukanlah masalah yang asing, Kredit merupakan salah satu pembiayaan sebagian besar dari kegiatan ekonomi dan sumber dana yang penting untuk setiap jenis usaha. Sebelum dimulainya pemberian kredit kepada nasabah diperlukan suatumengidentifikasi dan memprediksi
#  
# yang baik 
#  
# dan seksesama terhadap semua aspek  perkreditan yang dapat menunjang proses pemberian kredit, guna mencegah timbulnyamasalah resiko kredit.Setiap tahunnya industri perbankan mengalami peningkatan nasabah untuk kredit.Maka dari itu dalam menghadapi masalah resiko kredit yang dialami oleh Industri perbankansaat ini salah satunya dapat diatasi dengan mengidentifikasi dan memprediksi nasabahdengan baik sebelum memberikan pinjaman dengan cara memperhatikan data historis pinjaman. Oleh karena itu klasifikasi resiko kredit dalam perbankan memiliki peran yang penting. Apabila pengklasifikasian resiko kredit mengalami kesalahan, maka salah satudampak yang ditimbulkan adalah kredit macet. Kredit macet dapat menyebabkankebangkrutan pada bank
# 
# ## Import Package yang Dibutuhkan

# In[123]:


import numpy as np
import pandas as pd
from scipy import stats
from zipfile import ZipFile
import seaborn as sns
# !pip install imbalanced_learn
from imblearn.over_sampling import SMOTE
from collections import Counter
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc,classification_report, accuracy_score


# ## Load Dataset
# Informasi Dataset:
# 

# | Jenis | Keterangan | 
# | ----------- | :---------: | 
# | Sumber | https://www.kaggle.com/rikdifos/credit-card-approval-prediction | 
# | Kategori | Layak (0) dan Tidak Layak(1)| 

# In[2]:


zip_dir = "./Data/archive.zip"
zip_ref = ZipFile(zip_dir, 'r')
zip_ref
zip_ref.extractall()
zip_ref.close()


# In[3]:


app_df = pd.read_csv('./Data/application_record.csv')
credit_df = pd.read_csv('./Data/credit_record.csv')


# In[4]:


app_df.head()


# ## EDA ( Exploratory Data Analysis )

# ### Data Understanding

# In[5]:


# Memuat informasi dari dataset
app_df.info()


# In[6]:


# Melihat Statistik dataset
app_df.describe()


# In[7]:


# melihat jumlah nilai null pada dataset
app_df.isnull().sum()


# In[8]:


# Mengecek colom apa saja yang tidak mengandung nilai numerik
cat_columns = app_df.columns[(app_df.dtypes =='object').values].tolist()
cat_columns


# In[9]:


# Mengecek kolom apa saya yang mengandung nilai numerik
app_df.columns[(app_df.dtypes !='object').values].tolist()


# In[10]:


# mengecek nilai unik pada kolom non numerik

for i in app_df.columns[(app_df.dtypes =='object').values].tolist():
    print(i,'\n')
    print(app_df[i].value_counts())
    print('-----------------------------------------------')


# In[11]:


# mengecek nilai unik pada kolom numerik
app_df['CNT_CHILDREN'].value_counts()


# In[12]:


# Melihat nilai max dan min pada kolom "Days_Birth"
print('Min DAYS_BIRTH :', app_df['DAYS_BIRTH'].min(),'\nMax DAYS_BIRTH :', app_df['DAYS_BIRTH'].max())


# In[13]:


# Mengubah nilai kolom "Days_Birth" dari hari ke tahun
app_df['DAYS_BIRTH'] = round(app_df['DAYS_BIRTH']/-365,0)
app_df.rename(columns={'DAYS_BIRTH':'AGE_YEARS'}, inplace=True)


# In[14]:


# mengecek nilai unik pada kolom "Days_Employed" yang lebih besar dari 0
app_df[app_df['DAYS_EMPLOYED']>0]['DAYS_EMPLOYED'].unique()


# In[15]:


# Seperti disebutkan dalam dokumen, jika 'DAYS_EMPLOYED' positif tidak, berarti orang tersebut sedang menganggur, maka diganti dengan 0
app_df['DAYS_EMPLOYED'].replace(365243, 0, inplace=True)


# In[16]:


# Mengonversi nilai 'DAYS_EMPLOYED' dari Hari ke Tahun
app_df['DAYS_EMPLOYED'] = abs(round(app_df['DAYS_EMPLOYED']/-365,0))
app_df.rename(columns={'DAYS_EMPLOYED':'YEARS_EMPLOYED'}, inplace=True)  


# In[17]:


# Mengecek terdapat nilai apa saja pada kolom "Flag_Mobil"
app_df['FLAG_MOBIL'].value_counts()


# In[18]:


# Mengecek terdapat nilai apa saja pada kolom "FLAG_WORK_PHONE"
app_df['FLAG_WORK_PHONE'].value_counts()


# In[19]:


# Mengecek terdapat nilai apa saja pada kolom "FLAG_PHONE"
app_df['FLAG_PHONE'].value_counts()


# In[20]:


# Mengecek terdapat nilai apa saja pada kolom "FLAG_EMAIL"
app_df['FLAG_EMAIL'].value_counts()


# In[21]:


# Mengecek terdapat nilai apa saja pada kolom "CNT_FAM_MEMBERS"
app_df['CNT_FAM_MEMBERS'].value_counts()


# In[22]:


app_df.head()


# ### Data Cleaning
# 
# #### Menangani Missing Value
# Kolom yang akan kita hapus adalah :
# - occupation type : kolom ini dihapus karena banyak sekari terdapat missing value, jadi bisa dibilang kolom ini tidak memiliki peran penting terhadap prediksi yang akan dilakukan
# - Flag_mobil : Kolom ini dihapus karena hanya terdapat satu nilai saja
# - Flag_work_phone : Kolom ini hanya berisi nilai 0 & 1 untuk Seluler yang tidak dikirimkan, oleh karena itu hapus kolom
# - flag_phone : Kolom ini hanya berisi nilai 0 & 1 untuk Seluler yang tidak dikirimkan, oleh karena itu hapus kolom
# - flag_email : Kolom ini hanya berisi nilai 0 & 1 untuk email yang tidak dikirimkan, oleh karena itu hapus kolom

# In[23]:


app_df.drop('OCCUPATION_TYPE', axis=1, inplace=True)


# In[24]:


app_df.drop('FLAG_MOBIL', axis=1, inplace=True)


# In[25]:


app_df.drop('FLAG_WORK_PHONE', axis=1, inplace=True)


# In[26]:


app_df.drop('FLAG_PHONE', axis=1, inplace=True)


# In[27]:


app_df.drop('FLAG_EMAIL', axis=1, inplace=True)


# #### Menangani Outliers
# Outliers hanya terjadi pada data numerikal saja

# In[28]:


# Mengecek kolom yang berisi nilai numerik
app_df.columns[(app_df.dtypes !='object').values].tolist()


# In[29]:


num_cols = ['CNT_CHILDREN',
 'AMT_INCOME_TOTAL',
 'AGE_YEARS',
 'YEARS_EMPLOYED',
 'CNT_FAM_MEMBERS']

plt.figure(figsize=(19,9))
app_df[num_cols].boxplot()
plt.title("Numerical variables in the data", fontsize=20)
plt.show()


# In[30]:


# Fungsi Untuk Mendeteksi Outliers
def detect_outlier(data_1):
    outliers=[]
    threshold=3
    mean_1 = np.mean(data_1)
    std_1 =np.std(data_1)
    
    
    for y in data_1:
        z_score= (y - mean_1)/std_1 
        if np.abs(z_score) > threshold:
            outliers.append(y)
    return outliers


# In[31]:


AMT_outliers = detect_outlier(app_df['AMT_INCOME_TOTAL'])
CNTCh_outliers = detect_outlier(app_df['CNT_CHILDREN'])
YE_outliers = detect_outlier(app_df['YEARS_EMPLOYED'])
CFM_outliers = detect_outlier(app_df['CNT_FAM_MEMBERS'])
print("outliers")
print(f"AMT_INCOME_TOTAL : {len(AMT_outliers)}")
print(f"CNT_CHILDREN: {len(CNTCh_outliers)}")
print(f"YEARS_EMPLOYED : {len(YE_outliers)}")
print(f"CNT_FAM_MEMBERS : {len(CFM_outliers)}")


# In[32]:


# Fungsi untuk menghapus outliers
def remove_outlier(data):
    z = np.abs(stats.zscore(data))
    threshold = 3
    Q1 = np.percentile(data, 25,
                   interpolation = 'midpoint')
    Q3 = np.percentile(data, 75,
                   interpolation = 'midpoint')
    IQR = Q3 - Q1
    upper = data >= (Q3+1.5*IQR)
     # Below Lower bound
    lower = data <= (Q1-1.5*IQR)
    return data.index[upper]


# In[33]:


application_df = app_df.copy()
application_df.head()


# In[ ]:





# In[34]:


application_df.drop(remove_outlier(application_df["CNT_CHILDREN"]), inplace=True)
application_df.drop(remove_outlier(application_df["AMT_INCOME_TOTAL"]), inplace=True)
application_df.drop(remove_outlier(application_df["YEARS_EMPLOYED"]), inplace=True)
application_df.drop(remove_outlier(application_df["CNT_FAM_MEMBERS"]), inplace=True)


# In[35]:


application_df["AMT_INCOME_TOTAL"].value_counts()


# In[36]:


num_cols = ['CNT_CHILDREN',
 'AMT_INCOME_TOTAL',
 'AGE_YEARS',
 'YEARS_EMPLOYED',
 'CNT_FAM_MEMBERS']

plt.figure(figsize=(19,9))
application_df[num_cols].boxplot()
plt.title("Numerical variables in the data", fontsize=20)
plt.show()


# In[37]:


application_df.head()


# In[38]:


AMT_outliers = detect_outlier(application_df['AMT_INCOME_TOTAL'])
CNTCh_outliers = detect_outlier(application_df['CNT_CHILDREN'])
YE_outliers = detect_outlier(application_df['YEARS_EMPLOYED'])
CFM_outliers = detect_outlier(application_df['CNT_FAM_MEMBERS'])
print("outliers")
print(f"AMT_INCOME_TOTAL : {len(AMT_outliers)}")
print(f"CNT_CHILDREN: {len(CNTCh_outliers)}")
print(f"YEARS_EMPLOYED : {len(YE_outliers)}")
print(f"CNT_FAM_MEMBERS : {len(CFM_outliers)}")


# In[39]:


application_df.isnull().sum()


# #### Data Transforming

# Transforming Data credit_record.csv

# In[40]:


credit_df.head()


# In[41]:


credit_df['STATUS'].value_counts()


# In[42]:


# Mengkategorikan kolom 'STATUS' ke klasifikasi biner 0 : Klien Baik dan 1 : klien buruk
credit_df['STATUS'].replace(['C', 'X'],0, inplace=True)
credit_df['STATUS'].replace(['2','3','4','5'],1, inplace=True)
credit_df['STATUS'] = credit_df['STATUS'].astype('int')


# In[43]:


credit_df.info()


# In[44]:


credit_df['STATUS'].value_counts(normalize=True)*100


# In[45]:


credit_df_trans = credit_df.groupby('ID').agg(max).reset_index()


# In[46]:


credit_df_trans.drop('MONTHS_BALANCE', axis=1, inplace=True)
credit_df_trans.head()


# In[47]:


credit_df_trans['STATUS'].value_counts(normalize=True)*100


# ##### Merging Dataframe

# In[48]:


# menggabungkan dua set data berdasarkan 'ID'
final_df = pd.merge(application_df, credit_df_trans, on='ID', how='inner')
final_df.head()


# In[49]:


final_df.shape


# In[50]:


# menghapus kolom 'ID' karena hanya memiliki nilai unik (tidak diperlukan untuk Model ML)
final_df.drop('ID', axis=1, inplace=True)


# In[51]:


# menghapus record yang duplikat
final_df = final_df.drop_duplicates()
final_df.reset_index(drop=True ,inplace=True)


# In[52]:


final_df.shape


# In[53]:


final_df.isnull().sum()


# In[54]:


final_df['STATUS'].value_counts(normalize=True)*100


# In[55]:


# Mengonversi semua Kolom Non-Numerik ke Numerik
from sklearn.preprocessing import LabelEncoder

for col in cat_columns:
    if col != "OCCUPATION_TYPE":
        globals()['LE_{}'.format(col)] = LabelEncoder()
        final_df[col] = globals()['LE_{}'.format(col)].fit_transform(final_df[col])
final_df.head()  


# #### Data Visualization

# In[56]:


final_df.head()


# In[57]:


# Grafik ini menunjukkan bahwa, tidak ada kolom (Fitur) yang sangat berkorelasi dengan 'Status'
plt.figure(figsize = (8,8))
sns.heatmap(final_df.corr(), annot=True)
plt.show()


# In[58]:


# Grafik ini menunjukkan bahwa, sebagian besar aplikasi diajukan oleh Female's
plt.pie(final_df['CODE_GENDER'].value_counts(), labels=['Female', 'Male'], autopct='%1.2f%%')
plt.title('% of Applications submitted based on Gender')
plt.show()


# In[59]:


# Grafik ini menunjukkan bahwa, sebagian besar aplikasi disetujui untuk Wanita
plt.pie(final_df[final_df['STATUS']==0]['CODE_GENDER'].value_counts(), labels=['Female', 'Male'], autopct='%1.2f%%')
plt.title('% of Applications Approved based on Gender')
plt.show()


# In[60]:


# Grafik ini menunjukkan bahwa, mayoritas pemohon tidak memiliki mobil
plt.pie(final_df['FLAG_OWN_CAR'].value_counts(), labels=['No', 'Yes'], autopct='%1.2f%%')
plt.title('% of Applications submitted based on owning a Car')
plt.show()


# In[61]:


# Grafik ini menunjukkan bahwa, sebagian besar pemohon memiliki properti / Rumah Real Estate
plt.pie(final_df['FLAG_OWN_REALTY'].value_counts(), labels=['Yes','No'], autopct='%1.2f%%')
plt.title('% of Applications submitted based on owning a Real estate property')
plt.show()


# In[62]:


# Grafik ini menunjukkan bahwa, sebagian besar pelamar tidak memiliki anak
plt.figure(figsize = (8,8))
plt.pie(final_df['CNT_CHILDREN'].value_counts(), labels=final_df['CNT_CHILDREN'].value_counts().index, autopct='%1.2f%%')
plt.title('% of Applications submitted based on Children count')
plt.legend()
plt.show()


# In[63]:


# Grafik ini menunjukkan bahwa, sebagian besar pendapatan pemohon berkisar antara 100k hingga 300k
plt.hist(final_df['AMT_INCOME_TOTAL'], bins=20)
plt.xlabel('Total Annual Income')
plt.title('Histogram')
plt.show()


# In[64]:


# Grafik ini menunjukkan bahwa, sebagian besar pelamar bekerja secara profesional
plt.figure(figsize = (8,8))
plt.pie(final_df['NAME_INCOME_TYPE'].value_counts(), labels=final_df['NAME_INCOME_TYPE'].value_counts().index, autopct='%1.2f%%')
plt.title('% of Applications submitted based on Income Type')
plt.legend()
plt.show()


# In[65]:


# Grafik ini menunjukkan bahwa, mayoritas pelamar menyelesaikan Pendidikan Menengah
plt.figure(figsize=(8,8))
plt.pie(final_df['NAME_EDUCATION_TYPE'].value_counts(), labels=final_df['NAME_EDUCATION_TYPE'].value_counts().index, autopct='%1.2f%%')
plt.title('% of Applications submitted based on Education')
plt.legend()
plt.show()


# In[66]:


# Grafik ini menunjukkan bahwa sebagian besar pelamar sudah menikah
plt.figure(figsize=(8,8))
sns.barplot(final_df['NAME_FAMILY_STATUS'].value_counts().index, final_df['NAME_FAMILY_STATUS'].value_counts().values)
plt.title('% of Applications submitted based on Family Status')
plt.show()


# In[67]:


# Grafik ini menunjukkan bahwa, sebagian besar pemohon tinggal di Rumah/Apartemen
plt.figure(figsize=(12,5))
sns.barplot(final_df['NAME_HOUSING_TYPE'].value_counts().index, final_df['NAME_HOUSING_TYPE'].value_counts().values)
plt.title('% of Applications submitted based on Housing Type')
plt.show()


# In[68]:


# Grafik ini menunjukkan bahwa, mayoritas pelamar berusia 25 hingga 65 tahun
plt.hist(final_df['AGE_YEARS'], bins=20)
plt.xlabel('Age')
plt.title('Histogram')
plt.show()


# In[69]:


# Grafik ini menunjukkan bahwa, mayoritas pelamar Bekerja selama 0 hingga 7 tahun
plt.hist(final_df['YEARS_EMPLOYED'], bins=20)
plt.xlabel('No of Years Employed')
plt.title('Histogram')
plt.show()


# In[70]:


# Grafik ini menunjukkan bahwa, sebagian besar aplikasi ditolak jika Total pendapatan & tahun kerja kurang
sns.scatterplot(final_df['YEARS_EMPLOYED'], final_df['AMT_INCOME_TOTAL'], hue=final_df['STATUS'])
plt.title('Scatter Plot')
plt.show()


# ### Data Preprocessing

# In[71]:


final_df.head()


# #### Memisahkan Data menjadi Feature and Label

# In[72]:


cat_columns = final_df.columns[(final_df.dtypes =='object').values].tolist()
cat_columns


# In[73]:


for col in cat_columns:
    print(col , "  : ", globals()['LE_{}'.format(col)].classes_)


# In[74]:


final_df.corr()


# In[75]:


features = final_df.drop(['STATUS'], axis=1)
label = final_df['STATUS']


# In[76]:


features.head()


# In[77]:


label.head()


# #### Cek Imbalanced Data

# In[78]:


oversample = SMOTE(k_neighbors=5)
X_smote, Y_smote = oversample.fit_resample(features, label)


# In[79]:


counter = Counter(Y_smote)
print(counter)


# #### Normalisasi Data

# In[80]:


scaler_minmax =  MinMaxScaler()
# transform data
X_scaled = scaler_minmax.fit_transform(X_smote)
print(X_scaled)


# #### Split Data to Train and Test

# In[81]:


X_train,X_test,Y_train,Y_test = train_test_split(X_scaled, Y_smote, test_size = 0.2, random_state = 42)


# In[82]:


print(X_train.shape)
print(X_test.shape)


# ## Modeling
# ### Logistic Regression

# In[110]:


model = LogisticRegression(solver='liblinear', random_state=4)


# In[111]:


model.fit(X_train, Y_train)


# In[ ]:





# ### Evaluating Model

# In[112]:


y_pred = model.predict(X_test)


# In[113]:


print('Logistic Model Accuracy : ', model.score(X_test, Y_test)*100, '%')


# In[114]:


print(confusion_matrix(Y_test, y_pred))
print(classification_report(Y_test, y_pred))


# In[115]:


prob_estimates = model.predict_proba(X_test)

fpr, tpr, threshold = roc_curve(Y_test, prob_estimates[:,1])
nilai_auc = auc(fpr,tpr)

plt.plot(fpr, tpr, 'b', label=f'AUC={nilai_auc}')
plt.plot([0,1], [0,1], 'r--', label='Random Classifier')

plt.title('ROC: Reciever OPerating Characteristic')
plt.xlabel('Fallout or False Positive Rate')
plt.ylabel('Recal or True Positive Rate')
plt.legend()
plt.show()


# ### Tunning Hyperparameter

# In[116]:


#List Hyperparameters yang akan diuji
penalty = ['l1', 'l2']
C = np.logspace(-4,4,20)


# In[117]:


#Menjadikan ke dalam bentuk dictionary
hyperparameters = dict(penalty=penalty, C=C)


# In[118]:


#Memasukan ke Grid Search
#CV itu Cross Validation
#Menggunakan 10-Fold CV
clf = GridSearchCV(model, hyperparameters, cv=10)


# In[119]:


#Fitting Model
best_model = clf.fit(X_train, Y_train)


# In[120]:


#Nilai hyperparameters terbaik
print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
print('Best C:', best_model.best_estimator_.get_params()['C'])


# In[136]:


prob_estimates = best_model.predict_proba(X_test)

fpr, tpr, threshold = roc_curve(Y_test, prob_estimates[:,1])
nilai_auc = auc(fpr,tpr)

plt.plot(fpr, tpr, 'b', label=f'AUC={nilai_auc}')
plt.plot([0,1], [0,1], 'r--', label='Random Classifier')

plt.title('ROC: Reciever OPerating Characteristic')
plt.xlabel('Fallout or False Positive Rate')
plt.ylabel('Recal or True Positive Rate')
plt.legend()
plt.show()


# ### SVM

# In[124]:


svm_model = svm.SVC(kernel='linear')


# In[125]:


svm_model.fit(X_train, Y_train)


# In[126]:


y_pred_svm = svm_model.predict(X_test)


# In[128]:


print('SVM Model Accuracy : ', svm_model.score(X_test, Y_test)*100, '%')


# In[127]:


print(classification_report(Y_test, y_pred_svm))


# #### Tunning Hyperparameter

# In[129]:


param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']}


# In[130]:


grid = GridSearchCV(svm_model, param_grid, refit = True, verbose = 3)


# In[131]:


grid.fit(X_train, Y_train)


# In[132]:


print(grid.best_params_)


# In[133]:


print(grid.best_estimator_)


# In[134]:


grid_predictions = grid.predict(X_test)


# In[135]:


print(classification_report(Y_test, grid_predictions))


# #### Referensi
# - Ginting(2019). DATA MINING UNTUK ANALISA PENGAJUAN KREDIT DENGAN MENGGUNAKAN METODE LOGISTIK REGRESI. Jurnal Algoritma, Logika dan Komputasi, 164 - 169.
# - N Iriadi, H Leidiyana ( 2013 ).PREDIKSI PINJAMAN KREDIT DENGAN SUPPORT VECTOR MACHINE DAN K-NEAREST NEIGHBORS PADA KOPERASI SERBA USAHA
# 

# In[ ]:




