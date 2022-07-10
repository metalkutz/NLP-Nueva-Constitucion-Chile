# Análisis de la propuesta para la nueva Constitución de Chile
## Objetivo
El objetivo de este ejercicio es meramente exploratorio de los contenidos y temas tratados dentro del texto propuesto para la nueva Constitución de Chile. Se utilizan técnicas de análisis de NPL, utilizando librerias relacionadas. 

## Librerias utilizadas
- re
- pandas
- numpy
- wordcloud
- PIL
- nltk
- tabula (lectura pdf)
- sklearn (CountVectorizer, TfidfTransformer, TfidfCountVectorizer)
- matplotlib
- seaborn

## Fuente de información
Datos obtenidos desde sitio web oficial de la convención https://www.chileconvencion.cl/ y de la plataforma digital de participación popular https://plataforma.chileconvencion.cl/
Al inicio del análisis se revisó el borrador del texto de la nueva constitución que estaba disponible en pdf, junto con una estructuración de sus normas organizados por temática/comisión (json, csv). Luego en 7/julio/2022 se liberó el texto definitivo el cual es la propuesta para ser votado en el plebicito de salida en sept/2022. Los archivos estructurados separan los articulos y los clasifican en base a la comisión asociada. 

## Enfoque análisis
Se analizará el texto desde un punto de vista exploratorio cuantitativo. Sólo se presentarán resultados en base a los resultados que se encuentren con las distintas técnicas seleccionadas. 

- wordcloud de palabras
- riqueza lexica
- hapax de términos
- frecuencia de palabras
- bag of words de cada articulo
- td-idf  

![image](https://github.com/metalkutz/NPL-Nueva-Constitucion-Chile/blob/8d92f38790e0814525df213a6d37c6dc9ca98a22/constitucion/logo%20portada.png)