''' ANALISIS DEL BORRADOR DE LA NUEVA CONSTITUCIÓN DE CHILE
OBJETIVO: entender los principales conceptos que se tratan, descubrir los principales 
tópicos tratados, palabras utilizadas, y determinar si es posible entrenar modelos 
a) que logren agrupar los articulos similar al ámbito de la comisión al cual pertenece, 
b) que al introducir un articulo indique el ámbito de la comisión al cual pertenece.
1. análisis EDA
2. visualización
3. preprocesamiento
4. entrenamiento

'''

# %%
import pandas as pd
import numpy as np

import re
from wordcloud import WordCloud
from PIL import Image # tratamiento de imagenes
import PIL

import nltk
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt

# %%
'''datos obtenidos desde sitio web oficial de la convención 
https://www.chileconvencion.cl/
archivo viene en 3 formatos pdf, json, csv
'''
data = pd.read_csv(r'.\constitucion\normas_aprobadas_2022.06.30_11.16.csv')
data.info() # 7 columnas con los títulos de los articulos, comisión a la que pertenece,
# separamos el numero de la columna comision
data[['n_comision','nom_comision']] = data['Comisión'].str.split('-',expand=True) # separo el nombre de la comisión y su número en columnas distintas
data.head()
data.drop(['Comisión'], axis=1, inplace=True)
data.head()
# %%
#juntamos los articulos en 1 solo texto no sé porque XD
texto = data.Texto.sum()
# %%
'descargamos los stopwords en español'
nltk.download('stopwords') 
palabras_funcionales=nltk.corpus.stopwords.words("spanish")
palabras_funcionales.remove('estado')
# %%
'análisis EDA a través del uso del kit de NLTK'
nltk.download('spanish_grammars')
# %%
data["Texto_tk"] = data["Texto"].apply(lambda x:nltk.word_tokenize(x,"spanish"))
data["Texto_tk"] = data["Texto_tk"].apply(lambda x: list(map(str.lower, x)))
# %%
#eliminamos simbolos . , : ;
simbolos = ['.',',', ':' ,';','\'','(',')','&','#','%']
'quitamos los simbolos que no aportan en nada'
for i in simbolos:
    for j in data["Texto_tk"]:
        try:
            j.remove(i)
        except:
            continue

# %%
'wordcloud texto tokenizado sin simbolos'
texto_tk = data.Texto_tk.sum()
texto_const= ''
for i in texto_tk:
    texto_const=texto_const+' '+i

texto_const_wc = WordCloud(background_color='black', max_words=len(texto_const), stopwords=palabras_funcionales)
texto_const_wc.generate(texto_const)

fig, ax = plt.subplots(figsize=(10,10))
ax.imshow(texto_const_wc)
ax.axis('off')
plt.show()
# %%
'riqueza lexica baja 9%'
tokens = data.Texto_tk.sum()
palabras_totales = len(tokens)
tokens_conjunto=set(tokens) 
palabras_diferentes=len(tokens_conjunto)
riqueza_lexica=palabras_diferentes/palabras_totales
print('palabras totales',palabras_totales,'\n palabras diferentes',
    palabras_diferentes,'\n riqueza lexica',riqueza_lexica)

# %%
'dispersion palabra más repetidas y determino los hapaxes'
texto_nltk=nltk.Text(tokens) 
lista_palabras=["ley","derechos","podrá","personas","estado"] 
texto_nltk.dispersion_plot(lista_palabras)
# determino frecuencia de palababras
distribucion=nltk.FreqDist(texto_nltk)
lista_frecuencias=distribucion.most_common() 
# determino los hapax 
hapaxes=distribucion.hapaxes() 
for hapax in hapaxes: 
    print(hapax)
len(hapaxes)


# %%
'definimos una funcion para remover los stopwords de una lista de tokens'

def quitar_stopwords(tokens):
    stemmer = PorterStemmer()
    tokens_limpios=[]
    for token in tokens:
        if token not in palabras_funcionales:
            if len(token) > 1:
                if token != "...":
                    tokens_limpios.append(token)
    return [stemmer.stem(word) for word in tokens_limpios]
print(data["Texto_tk"][0])

data["Texto_tk_limpio"] = data["Texto_tk"].apply(lambda x: quitar_stopwords(x))
# %%
tokens_limpios = data.Texto_tk_limpio.sum()
nltk.FreqDist(nltk.Text(tokens_limpios)).plot(10)
# %%
'wordcloud de palabras sin stopwords y stemizado'
texto_tk_limpio = data.Texto_tk_limpio.sum()
texto_const_limpio= ''
for i in texto_tk:
    texto_const_limpio=texto_const_limpio+' '+i

texto_const_limpio_wc = WordCloud(background_color='black', max_words=len(texto_const_limpio), stopwords=palabras_funcionales)
texto_const_wc.generate(texto_const_limpio)

fig, ax = plt.subplots(figsize=(10,10))
ax.imshow(texto_const_wc)
ax.axis('off')
plt.show()
# %%
import sklearn.feature_extraction.text as txt
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfTransformer, TfidfCountVectorizer

# %%
'armamos el bag of words'
#convertimos en frase la lista de tokens para generar el vector
data['Texto_frase_limpia'] = data["Texto_tk_limpio"].apply(lambda x: ' '.join(x)).values
count = txt.CountVectorizer()
array=data["Texto_frase_limpia"]
bag = count.fit_transform(array)

tfidf = txt.TfidfTransformer()
np.set_printoptions(precision=2)
print(tfidf.fit_transform(bag).toarray().shape)
# %%
txt.TfidfCountVectorizer()
# %%
