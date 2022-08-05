# Análisis Cuantitativo Artículos Nueva Constitución
Para el analisis del texto se realiza un tratamiento previo de las palabras de tal forma faciliar la interpretación y normalizar/igualar palabras que permitan que la agregación sea más representativa y muestre de mejor manera el uso de palabras. 

Para ello es que utilizamos herramientas de expresión regular y funciones de procesamiento de lenguaje natural (librería NLTK): 
- transformación letras a minúsculas
- eliminación de tildes
- eliminación de caracteres especiales
- eliminación de stopwords o palabras funcionales que no son relevantes (ej. artículos)
- transformación de palabras a su raiz (stemming), de esta forma palabras que tienen el mismo significado pero que estén en distinto tiempo verbal o similar, se pueden agrupar    

## Texto Completo
---
Antes de realizar el tratamiento de lenguaje natural sobre el texto, se realiza un análisis del total de palabras y riqueza léxica. 

**palabras totales** = 5.1106 

**palabras únicas** = 4.995 

**riqueza léxica** = 0.098

Además podemos determinar cuales son las palabras que aparecen solo 1 vez en el texto: 

tamaños lista hapaxes=2.042

Algunos ejemplos: 
regresivo
disminuya
menoscabe
injustificadamente
arbitrariamente
restringida
sorprendida
deudas
remediacion
cuerpo
placer
anticoncepcion
gestar
interrupcion
voluntarios
violencias
interferencias
individuos
beneficiarse
cientifico
discriminatoria ... para listado completo referirse a archivo hapax.txt

Ahora procesamos las palabras del texto aplicando la herramientas de la libreria NLTK (salvo stemming) para representar la frecuencia de palabras del texto en distintas vistas. 

**Wordcloud Texto Completo** 
![](wordcloud%20token.png)

A partir de las palabras con mayor frecuencia se grafica la dispersión de frecuencia de palabras y su dispersión léxica dentro del texto. 

**Gráfico Dispersion frecuencia top10 palabras**

![](dispersion%20frecuencia%20palabras-top10.png)

**Gráfico Dispersión Léxica**

![](graf%20disp%20lexico.png)

Luego analizamos las palabras aplicando stemming

**Wordcloud Texto Completo con stemming** 



**Gráfico Dispersion frecuencia top10 palabras stemming**
![](dispersion%20frecuencia%20palabras%20st-top10.png)


**Gráfico Dispersión Léxica stemming**


