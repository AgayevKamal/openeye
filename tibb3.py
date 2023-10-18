# Lazım olan kitabxanalar
from sklearn.base import BaseEstimator, TransformerMixin
from PIL import Image
import streamlit as st
import plotly.express as px
import pandas as pd
import warnings
import pickle
import time
import PIL

# Potensial xəbərdarlıqların filterlənməsi
warnings.filterwarnings(action = 'ignore')

# Datasetin yüklənməsi
df = pd.read_csv('cleaned_data.csv')
    

# İlkin əməliyyatlardan ibarət sinifin yaradılması
class InitialPreprocessor(BaseEstimator, TransformerMixin):
    # fit funksiyasının yaradılması
    def fit(self, X, y = None):
        # Nominal dəyişənlərdən ibarət list data strukturunun yaradılması
        self.nominal_features = ['şəhər', 'marka', 'model', 'ban_növü', 'rəng', 'sürətlər_qutusu',
       'ötürücü', 'yeni', 'hansı_bazar_üçün_yığılıb', 'yanacaq_novu',
       'vuruğu_var', 'rənglənib', 'lehimli_disk', 'abs', 'lyuk',
       'yağış_sensoru', 'mərkəzi_qapanma', 'park_radarı', 'kondisioner',
       'oturacaqların_isidilməsi', 'dəri_salon', 'ksenon_lampalar',
       'arxa_görüntü_kamerası', 'yan_pərdələr',
       'oturacaqların_ventilyasiyası']
        
        # Listlərin geri qaytarılması
        return self
    
    # transform funksiyasının yaradılması
    def transform(self, X, y = None):
        # Nominal dəyişənlərdə ola biləcək potensial boşluqların silinib bütün dəyərlərin ilk hərflərinin böyüdülməsi
        X[self.nominal_features] = X[self.nominal_features].applymap(func = lambda x: x.strip().capitalize(), na_action = 'ignore')
        # Əməliyyat tətbiq olunmuş asılı olmayan dəyişənlərin geri qaytarılması
        return X


interface = st.container()



with interface:
    
    st.title(body = 'Moto4 Prediction System')
    
