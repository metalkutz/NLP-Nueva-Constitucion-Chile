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

![](wordcloud%20token%20limpio%20stem.png)

**Gráfico Dispersion frecuencia top10 palabras stemming**
![](dispersion%20frecuencia%20palabras%20st-top10.png)


**Gráfico Dispersión Léxica stemming**

![](graf%20disp%20lexico%20tokens%20stem%20sin%20sw.png)

## Comisiones
---
Se analiza de forma separada los artículos redactados por las distintas comisiones
- 1: ' Sistemas de Conocimientos, Culturas, Ciencia, Tecnología, Artes y Patrimonios'
- 2: ' Derechos de los Pueblos Indígenas y Plurinacional  '   
- 3: ' Sistema de Justicia, Órganos Autónomos de Control y Reforma Constitucional'
- 4: ' Derechos Fundamentales'
- 5: ' Sobre Principios Constitucionales, Democracia, Nacionalidad y Ciudadanía'
- 6: ' Forma de Estado, Ordenamiento, Autonomía, Descentralización, Equidad, Justicia Territorial, Gobiernos Locales y Organización Fiscal'
- 7: ' Medio Ambiente, Derechos de la Naturaleza, Bienes Naturales Comunes y Modelo Económico'
- 9: ' Sobre Sistema Político, Gobierno, Poder Legislativo y Sistema Electoral'

### Comision 1: Sistemas de Conocimientos, Culturas, Ciencia, Tecnología, Artes y Patrimonios
![](wc_com1_conoc.png)
![](disp_frec_palab_com1.png)

### Comision 2: Derechos de los Pueblos Indígenas y Plurinacional
![](wc_com2_derechospueblosorigin.png)
![](disp_frec_palab_com2.png)

### Comision 3: Sistema de Justicia, Órganos Autónomos de Control y Reforma Constitucional
![](wc_com3_justi.png)
![](disp_frec_palab_com3.png)
### Comision 4: Derechos Fundamentales
![](wc_com4_derechosfunda.png)
![](disp_frec_palab_com4.png)
### Comision 5: Sobre Principios Constitucionales, Democracia, Nacionalidad y Ciudadanía
![](wc_com5_princip.png)
![](disp_frec_palab_com5.png)
### Comision 6: Forma de Estado, Ordenamiento, Autonomía, Descentralización, Equidad, Justicia Territorial, Gobiernos Locales y Organización Fiscal
![](wc_com6_formaestad.png)
![](disp_frec_palab_com6.png)
### Comision 7: Medio Ambiente, Derechos de la Naturaleza, Bienes Naturales Comunes y Modelo Económico
![](wc_com7_medioamb.png)
![](disp_frec_palab_com7.png)
### Comision 9: Sobre Sistema Político, Gobierno, Poder Legislativo y Sistema Electoral
![](wc_com9_sistema.png)
![](disp_frec_palab_com9.png)

## Clusters de palabras
Análisis exploratorio utilizando distintas técnicas de clustering para determinar si existen patrones dentro del texto en base a las palabras utilizadas. 

### Clase 0
![](wc_kmeans_c0.png)
![](disp_frec_palab_kmeans0.png)
### Clase 1
![](wc_kmeans_c1.png)
![](disp_frec_palab_kmeans1.png)
### Clase 2
![](wc_kmeans_c2.png)
![](disp_frec_palab_kmeans2.png)
### Clase 3
![](wc_kmeans_c3.png)
![](disp_frec_palab_kmeans3.png)
### Clase 4
![](wc_kmeans_c4.png)
![](disp_frec_palab_kmeans4.png)
### Clase 5
![](wc_kmeans_c5.png)
![](disp_frec_palab_kmeans5.png)
### Clase 6
![](wc_kmeans_c6.png)
![](disp_frec_palab_kmeans6.png)