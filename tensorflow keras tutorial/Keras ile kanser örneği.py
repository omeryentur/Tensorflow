# -*- coding: utf-8 -*-

# =============================================================================
# Gerekli kütüphaneleri içe aktaralım
# =============================================================================
from tensorflow.keras import layers
import tensorflow as tf
from sklearn.datasets import load_breast_cancer #veri seti için

# =============================================================================
# DATA SETİMİZİ ÇEKELİM
# =============================================================================
x_train ,y_train =load_breast_cancer(return_X_y=True) 

# =============================================================================
# Hiper parametreli atayalım
# =============================================================================

input_size=30            #giriş boyutunu atadık
hidden_size=128          #gizli katman boyutu
output_size=2            #çıkış boyutumuzu atadık
learning_rate=0.01      #öğrenmek katsayımızı belirledik
EPOCHS= 100            #tekrarlama sayımızı atadık

model=tf.keras.Sequential()  #model başlatalım

model.add(layers.Dense(input_size, activation='relu')) # 30 giriş boyutu oluşturalım ve relu aktivasyon fonksiyonu işleme sokalım
model.add(layers.Dense(hidden_size, activation='relu')) # 128 giriş boyutu oluşturalım ve relu aktivasyon fonksiyonu işleme sokalım
model.add(layers.Dense(output_size)) # iki çıkışımız olduğu için çıkş olarak 2 yaptık

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),  #adam optimizer öğrenme katsayısı =0.01
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), # iki şeçim için SparseCategoricalCrossentropy 
              metrics=['accuracy'])

model.fit(x_train,y_train,epochs=EPOCHS,validation_split=0.17)        # modeli eğittelim  eğittim için verini %83 mü validation içinde %17 kullandık
model.summary()                                 # modelin  parametre sayısı , katmanlar ile ilgili özet
