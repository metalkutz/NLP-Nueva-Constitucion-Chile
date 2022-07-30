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

from funciones import *

####### BAG OF WORDS: normalización vector de palabras #######
from sklearn.feature_extraction.text import  TfidfVectorizer

###### Graficos y visualizacion ########
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from PIL import Image # tratamiento de imagenes
import PIL
import seaborn as sns

###### PCA #######
from sklearn.decomposition import PCA

##### Clustering #####
from sklearn.cluster import DBSCAN , KMeans , AgglomerativeClustering
from sklearn.datasets import make_circles
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from scipy.spatial import distance_matrix 
from scipy.cluster import hierarchy 

sw.remove('estado')  # removemos la palabra estado de la lista de stopwords
# %%
'''datos obtenidos desde sitio web oficial de la convención 
https://www.chileconvencion.cl/
archivo viene en 3 formatos pdf, json, csv
'''
data = pd.read_csv(r'.\constitucion\normas_aprobadas_2022.06.30_11.16.csv')
data.info() # 7 columnas con los títulos de los articulos, comisión a la que pertenece,
data.head()
# %%
data.Comisión.unique()
# %%
# separamos el numero de la columna comision
data[['n_comision','nom_comision']] = data['Comisión'].str.split('-',expand=True) # separo el nombre de la comisión y su número en columnas distintas
data.head()
data.drop(['Comisión','Aprobada en'], axis=1, inplace=True)
data.head()
# %%
'#### Análisis inicial sobre comisiones'
data.groupby(['n_comision']).size()/len(data) # comision 6, 1 y 3 representan el 63% de los articulos
# %%
print(data.groupby(['n_comision','nom_comision']).size()/len(data))
# las comisiones con menos artículos son pueblos indígeneas, medio ambiente y cultura/ciencias/tech
# %%
print(data['Cantidad de Comentarios'].unique())
data.drop(['Cantidad de Comentarios'], axis=1, inplace=True)
# %%
'analizamos el texto completo sin limpieza de palabras'
#juntamos los articulos en 1 solo texto
texto = data.Texto.sum()
tokens = nltk.word_tokenize(texto,"spanish")

'riqueza lexica 10.7%'
#tokens = data.Texto_tk.sum()
palabras_totales = len(tokens) # calculo el total de palabras del texto
palabras_unicas = set(tokens)  # creo objeto set de datos para tener palábras únicas
palabras_diferentes = len(palabras_unicas) # calculo el total de palabras diferentes a partir del set de palabras
riqueza_lexica = palabras_diferentes/palabras_totales # determino la riqueza léxica
print('palabras totales',palabras_totales,'\n palabras diferentes',
    palabras_diferentes,'\n riqueza lexica',riqueza_lexica)

# %%
'#### aplicamos las transformaciones para limpiar y generar tokens'
df1 = data.copy()
df1['Texto_limpio']= df1['Texto'].astype(str)
df1['Texto_limpio'] = df1['Texto'].apply(lambda texto: texto_limpio(texto)) 
df1["Texto_limpio_tk"] = df1["Texto_limpio"].apply(lambda x:nltk.word_tokenize(x,"spanish"))
# df1["Texto_tk"] = df1["Texto_tk"].apply(lambda x: list(map(str.lower, x)))
df1.head()

# %%
'wordcloud texto tokenizado sin simbolos ni numeros'
Texto_limpio_tk = df1.Texto_limpio_tk.sum()
texto_const= ''
for i in Texto_limpio_tk:
    texto_const=texto_const+' '+i

texto_const_wc = WordCloud(background_color='black', max_words=len(texto_const), stopwords=sw)
texto_const_wc.generate(texto_const)

fig, ax = plt.subplots(figsize=(20,20))
ax.imshow(texto_const_wc)
ax.axis('off')
plt.show()
# %%
'riqueza lexica baja 9.9%'
#tokens = df1.Texto_tk.sum()
palabras_totales = len(Texto_limpio_tk) # calculo el total de palabras del texto
palabras_unicas = set(Texto_limpio_tk)  # creo objeto set de datos para tener palábras únicas
palabras_diferentes = len(palabras_unicas) # calculo el total de palabras diferentes a partir del set de palabras
riqueza_lexica = palabras_diferentes/palabras_totales # determino la riqueza léxica
print('palabras totales',palabras_totales,'\n palabras diferentes',
    palabras_diferentes,'\n riqueza lexica',riqueza_lexica)

# %%
'dispersion palabra más repetidas y determino los hapaxes'
texto_nltk = nltk.Text(Texto_limpio_tk) # lo convierto a objeto texto de nltk

# determino frecuencia de palababras
distribucion = nltk.FreqDist(texto_nltk) # calculo la distribución de la frecuencia de palabras dentro del texto
lista_frecuencias = distribucion.most_common() # mismo cálculo de distribución pero como tipo de objeto lista
lista_frecuencias
# %%
nltk.FreqDist(nltk.Text(Texto_limpio_tk)).plot(20)

# %%
lista_palabras = ["ley","derechos","estado","constitucion","derecho"] # lista de palabras más repetidas 
texto_nltk.dispersion_plot(lista_palabras) # grafico la dispersión dentro del texto de las palabras más repetidas

# %%
# determino los hapax 
hapaxes = distribucion.hapaxes() # buscamos los hapaxses sobre el objeto nltk de distribución de palabras 
for hapax in hapaxes: 
    print(hapax)
print('tamaños lista hapaxes:',len(hapaxes))#2042
print('tamaño lista frecuencia palabras:',len(lista_frecuencias),'x',len(lista_frecuencias[0]))
# %%
'contexto de la palabra ser'
texto_nltk.concordance("ser")

# %%
# creamos nueva columna sobre datos limpios, ahora sin stopwords
df1["Texto_tk_sw"] = df1["Texto_limpio_tk"].apply(lambda x: quitar_sw(x))
# stemizamos las palabras y guardamos en nueva columna
df1["Texto_tk_sw_st"] = df1["Texto_tk_sw"].apply(lambda x: stemizar_raiz(x))
# lematizamos las palabras y guardamos en nueva columna
df1["Texto_tk_sw_lm"] = df1["Texto_tk_sw"].apply(lambda x: lematizar(x))
df1.head()
# %%
# repetimos ejercicio ahora sobre columna sin stopwords y graficamos frecuencia
texto_tk_sw = df1.Texto_tk_sw.sum()
texto_tk_sw_st = df1.Texto_tk_sw_st.sum()
texto_tk_sw_lm = df1.Texto_tk_sw_lm.sum()
nltk.FreqDist(nltk.Text(texto_tk_sw)).plot(10)
nltk.FreqDist(nltk.Text(texto_tk_sw_st)).plot(10)
nltk.FreqDist(nltk.Text(texto_tk_sw_lm)).plot(10)

texto_const_st= ''
for i in texto_tk_sw_st:
    texto_const_st=texto_const_st+' '+i

texto_const_lm= ''
for i in texto_tk_sw_lm:
    texto_const_lm=texto_const_lm+' '+i

# %%
'wordcloud de palabras sin stopwords y stemizado'
texto_const_st_wc = WordCloud(background_color='black', max_words=len(texto_const_st), stopwords=sw)
texto_const_st_wc.generate(texto_const_st)

fig, ax = plt.subplots(figsize=(10,10))
ax.imshow(texto_const_st_wc)
ax.axis('off')
plt.show()
# %%
'wordcloud de palabras sin stopwords y lematizado'
texto_const_lm_wc = WordCloud(background_color='black', max_words=len(texto_const_lm), stopwords=sw)
texto_const_lm_wc.generate(texto_const_lm)

fig, ax = plt.subplots(figsize=(10,10))
ax.imshow(texto_const_lm_wc)
ax.axis('off')
plt.show()
# %%
'dispersion palabra stemizadas más repetidas y determino los hapaxes'
texto_nltk_st = nltk.Text(texto_tk_sw_st) # lo convierto a objeto texto de nltk
# determino frecuencia de palababras
distribucion = nltk.FreqDist(texto_nltk_st) # calculo la distribución de la frecuencia de palabras dentro del texto
lista_frecuencias = distribucion.most_common() # mismo cálculo de distribución pero como tipo de objeto lista
lista_frecuencias
# %%
lista_palabras = ["ley","derech","estad",'ser','public','constitucion','podr','person','diput','regional','deber'] # lista de palabras más repetidas 
texto_nltk_st.dispersion_plot(lista_palabras) # grafico la dispersión dentro del texto de las palabras más repetidas

# %%
# determino los hapax 
hapaxes = distribucion.hapaxes() # buscamos los hapaxses sobre el objeto nltk de distribución de palabras 
for hapax in hapaxes: 
    print(hapax)
len(hapaxes) #2042
# %%
'########## Vectorizamos ponderando peso de palabras por articulo ####################'
df1['Articulo_limpio'] = df1["Texto_tk_sw_st"].apply(lambda x: ' '.join(x)).values #convertimos en frase la lista de tokens para generar el vector
'armamos el bag of words'
bag_articulos_limpios = np.array(df1["Articulo_limpio"]) # convertimos token a array de palabras
np.set_printoptions(precision=2)
vectorizador = TfidfVectorizer() #creamos objeto tf-idf
matriz_palabras = vectorizador.fit_transform(bag_articulos_limpios) #creamos el bag of words normalizado
matriz_palabras = matriz_palabras.astype('float32')
# %%
matriz_palabras.toarray().shape

# %%
n_palabras= vectorizador.vocabulary_
n_palabras
'armamos la matriz con el objeto vectorizador tfidf'
df2 = pd.DataFrame(matriz_palabras.toarray())  # el array de matriz palabras pasamos a dataframe
df2.columns = vectorizador.get_feature_names_out() # agregamos nombres a las columnas con las palabras del vocabulario
df2.head()
nom_df2_colum = df2.columns
nom_df2_colum

# %%
'############### PCA no me funciona con los pocos registros que hay...  #############'
pca = PCA() # objeto de PCA n_components=1000 con un máximo de 1000 componentes
pca = pca.fit(df2) # ajustamos el PCA al df2 de matriz de palabras

print(pca.components_.shape)

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('Explained variance')
plt.show()

lista_PCA = [ 'PC'+str(i) for i in range(len(pca.components_)) ] # generamos la lista de nombres de componentes del PCA

reduced_data = pca.transform(df2)  # aplicamos la transformación al dataframe de la matriz de palabras reduciendo la dimensionalidad
reduced_data = pd.DataFrame(reduced_data, columns = lista_PCA) # agregamos nombre de las columnas asociadas a los componentes del PCA

# %%
#temp = data.reset_index() #df original reseteamos el indice para poder concatenar
# df3 = pd.concat([temp,reduced_data], axis=1) #concatenamos dataframe original con componentes
df3 = pd.concat([df1,df2], axis=1)
df3.head()

# %%
'############ datos para train y test ###########'
#X = df3.drop(columns=['label'], axis=1) # creamos las variables independientes
#y = df3['label']  # creamos la variable dependiente
X = df2.values
# %%
'primero determinamos el valor de epsilon'
neigh = NearestNeighbors(n_neighbors=2)
nbrs = neigh.fit(X)
distances, indices = nbrs.kneighbors(X)

distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.plot(distances)
# 1.25
# %%
epsilon = 1.25
minimumSamples = 2
db = DBSCAN(eps=epsilon, min_samples=minimumSamples).fit(X)
labels_dbscan = db.labels_

df2["clase-dbscan"]=labels_dbscan
df3["clase-dbscan"]=labels_dbscan
df1["clase-dbscan"]=labels_dbscan

plt.figure(figsize=(8,6))
#sns.scatterplot(df2['ley'] , df2['derech'], hue = df2['clase-dbscan'], palette="Set2")
sns.scatterplot(x='ley' , y='derech', hue = 'clase-dbscan', data=df2, palette="Set2")

plt.show()
# %%
'K-means'
# graficamos la inercia para determinar el codo
inercias = [] 
  
for k in range(2,20): 
    kmeans = KMeans(k)
    kmeans.fit(X)     
    inercias.append(kmeans.inertia_) 
inercias

plt.plot(range(2,20), inercias, 'bx-') 
plt.xlabel('Ks') 
plt.ylabel('Inercia') 
plt.show()
# %%
# ahora graficamos la metrica silhouette 
silhouette_coefficients = []
for k in range(2, 20):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    score = silhouette_score(X, kmeans.labels_)
    silhouette_coefficients.append(score)

plt.plot(range(2, 20), silhouette_coefficients)
plt.xticks(range(2, 20))
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Coefficient")
plt.show()
# definimos el número de cluster sobre el valor más alto de la silhouette  
# %%
###### 7 clusters la 
kmeans = KMeans(n_clusters=7)
kmeans.fit(X)
kmeans.predict(X)

### calculo los centroides
kmeans.cluster_centers_
df_centroides  = pd.DataFrame(data = kmeans.cluster_centers_, columns = nom_df2_colum)
print(df_centroides)

df2['clase-kmeans'] = kmeans.labels_
df1["clase-kmeans"] = kmeans.labels_
df3["clase-kmeans"] = kmeans.labels_
plt.figure(figsize=(8,6))
sns.scatterplot(df2['ley'] , df2['derech'], hue = df2['clase-kmeans'], palette="Set2")
plt.show()
# %%
df_centroides
# %%
'cluster jerarquico'
# construyo matriz de distancias
dist_matrix = distance_matrix(X,X)
# elegimos el tipo de distancia 
Z = hierarchy.linkage(dist_matrix, 'complete') 

# graficamos el dendograma
plt.figure(figsize=(20,10))
dendro = hierarchy.dendrogram(Z)
plt.tick_params(axis='x', labelsize=8)
# %%
# elegimos el numero de clusters para aglomerar
agglom = AgglomerativeClustering(n_clusters = 7, linkage = 'complete')
agglom.fit(X)
df2['clase-jerarquico'] = agglom.labels_
df3['clase-jerarquico'] = agglom.labels_
df1['clase-jerarquico'] = agglom.labels_

plt.figure(figsize=(8,6))
sns.scatterplot(df2['ley'] , df2['estad'], hue = df2['clase-jerarquico'], palette="Set2")
plt.show()
# %%
df3.head()
# %%
'''
##### Analisis de clusters
'''
df3.groupby(['clase-dbscan']).size()/len(df3) # no sirve, clase 0 se lleva el 86% de los articulos y 10% considera outliers
df3.groupby(['clase-jerarquico']).size()/len(df3) # gran concentración en 1 cluster 0 con 47% de los articulos, luego sigue 1 con 23% y luego 2 con 14% 
df3.groupby(['clase-kmeans']).size()/len(df3) # gran concentración en 2 clusters 0 y 5 suman 60% de la muestra, resto de clusters se reparte los artículos de forma similar con menos de 10%
# %%
'######## analisis kmeans'
df_kmeans = df3.copy()
df_kmeans.drop(['Aprobada en','Cantidad de Comentarios','URL','clase-dbscan','clase-jerarquico'], axis=1, inplace=True)
# %%
'eliminamos las palabras con mayor frecuencia'
eliminar_palabras = ['ley','estado','constitución','derecho','derechos','podra','personas','ejercicio']
eliminar_palabras_st = ['ley','estado','constitución','derecho','derechos','podra','personas','ejercicio']
sw_clusters = sw.copy()
sw_clusters.extend(eliminar_palabras) 
# %%
# 7 clases 
clase=0
df_kmeans0 = df_kmeans[df_kmeans['clase-kmeans']==clase].groupby('n_comision').size()/len(df_kmeans[df_kmeans['clase-kmeans']==clase])

texto_tk_sw = df_kmeans.Texto_tk_sw.sum()
texto_tk_sw_st = df_kmeans.Texto_tk_sw_st.sum()
nltk.FreqDist(nltk.Text(texto_tk_sw)).plot(10)
nltk.FreqDist(nltk.Text(texto_tk_sw_st)).plot(10)

texto_const_st= ''
for i in texto_tk_sw_st:
    texto_const_st=texto_const_st+' '+i
# %%
'wordcloud de palabras sin stopwords y stemizado'
texto_const_st_wc = WordCloud(background_color='black', max_words=len(texto_const_st), stopwords=sw)
texto_const_st_wc.generate(texto_const_st)

fig, ax = plt.subplots(figsize=(10,10))
ax.imshow(texto_const_st_wc)
ax.axis('off')
plt.show()