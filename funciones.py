####### NPL ########
import re
import nltk
#from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer

nltk.download('spanish_grammars')

'descargamos los stopwords en español'
nltk.download('stopwords')  # descargamos los stop words en español
sw = nltk.corpus.stopwords.words("spanish")
# sw.remove('estado')  # removemos la palabra estado de la lista de stopwords
# sw.remove("no")

lemmatizer = WordNetLemmatizer()
stemmer = SnowballStemmer("spanish")
#stemmer = PorterStemmer()

'###### Funciones para de tratamiento de texto'

# función para quitar tildes
def sin_tildes(s):
    tildes = (
        ("á", "a"),
        ("é", "e"),
        ("í", "i"),
        ("ó", "o"),
        ("ú", "u"),
    )
    for origen, destino in tildes:
        s = s.replace(origen, destino)
    return s

# funcion para eliminar caracteres especiales, convierte todo a minúscula, quita tildes 
def texto_limpio(texto):
    texto = texto.lower() # convertir en minúsculas
    texto = re.sub(r"[\W\d_]+", " ",texto) # remover caract especiales y números
    texto = sin_tildes(texto) # remove tildes
    return texto

def textotk_palabracorta(lista_tk):
    #texto = texto.split() # tokenizar
    lista_tk = [palabra for palabra in lista_tk if len(palabra) > 2] # eliminar palabras con menos de 3 letras
    #texto = " ".join(texto)
    return lista_tk

def quitar_sw(lista_tk):
    #texto = texto.split() # tokenizar
    lista_tk = [palabra for palabra in lista_tk if palabra not in sw] # stopwords
    #texto = " ".join(texto)
    return lista_tk

def stemizar_raiz(lista_tk):
    #texto = texto.split() # tokenizar
    lista_tk = [stemmer.stem(palabra) for palabra in lista_tk] #stem
    #texto = " ".join(texto)
    return lista_tk

def lematizar(lista_tk):
    #texto = texto.split() # tokenizar
    lista_tk = [lemmatizer.lemmatize(palabra) for palabra in lista_tk] #stem
    #texto = " ".join(texto)
    return lista_tk

# funcion para remover los stopwords y stemizar palabras tokenizadas dentro de una lista
def quitar_stopwords(lista_tk):
    stemmer = PorterStemmer()
    tokens_limpios=[]
    for token in lista_tk:
        if token not in sw:
            if len(token) > 2:
                if token != "...":
                    tokens_limpios.append(token)
    return [stemmer.stem(word) for word in tokens_limpios]