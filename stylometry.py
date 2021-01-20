from collections import Counter
import numpy as np
from matplotlib import pyplot as plt
import re, itertools
from sklearn.model_selection import train_test_split
from termcolor import colored

#from keras.utils import to_categorical


def hscore(co,s1,s2):
    try:
        h = sum([co.get(t,0) for t in s1.split()])/\
            sum([co.get(t,0) for t in s2.split()])
    except:
        h=0
    return h

def horton_discriminators(text):
    text=text.lower()
    for k in '.,!?\'\";:()[]{}-':
        text=text.replace(k,' ')
    co = Counter(text.split())
    f1 = hscore(co,'did','did do')
    T10='but by for no not so that the to with'
    f2 = hscore(co,'no',T10)
    f3 = hscore(co,'no','no not')
    f4 = hscore(co,'the','to')
    f5 = hscore(co,'upon','on upon')
    return [f1,f2,f3,f4,f5]

def clean(text):
    text = text.replace('\n',' ')
    T = text.split('gutenberg')
    text = T[np.argmax([len(t) for t in T])]
    text=text.lower()
    for k in '.,!?\'\"”;:-=_~#^&%$*@/|()[]{}<>1234567890':
        text=text.replace(k,' ')
    text = re.sub('\s+',' ',text)
    return text

def bigrams(words):
    return [i+' '+j for i,j in zip(words[:-1],words[1:])]

def trigrams(words):
    return [i+' '+j+' '+k for i,j,k
            in zip(words[:-2],words[1:-1],words[2:])]

def generate_features(corpus1,corpus2,kf=500):
    biwords1 = bigrams(corpus1.split())
    c1 = Counter(biwords1)
    c1s = sum(c1.values())
    c1 = {c:g/c1s for c,g in c1.items()}

    biwords2 = bigrams(corpus2.split())
    c2 = Counter(biwords2)
    c2s = sum(c2.values())
    c2 = {c:a/c2s for c,a in c2.items()}
    
    disc={}
    for w in set(c2)&set(c1):
        disc[w]=(c2.get(w,0)-c1.get(w,0))*\
               abs((c2.get(w,0)-c1.get(w,0)))/\
               (c2.get(w,0)+c1.get(w,0))**0
        
    most_disc=Counter(disc).most_common()
    features = [c[0] for c in most_disc[:kf]+most_disc[-kf:]]
    return features

def generate_features_from_words(words1,words2,kf=500):
    c1 = Counter(words1)
    c1s = sum(c1.values())
    c1 = {c:g/c1s for c,g in c1.items()}

    c2 = Counter(words2)
    c2s = sum(c2.values())
    c2 = {c:a/c2s for c,a in c2.items()}
    
    disc={}
    for w in set(c2)&set(c1):
        disc[w]=(c2.get(w,0)-c1.get(w,0))*\
               abs((c2.get(w,0)-c1.get(w,0)))/\
               (c2.get(w,0)+c1.get(w,0))**0
        
    most_disc=Counter(disc).most_common()
    features = [c[0] for c in most_disc[:kf]+most_disc[-kf:]]
    return features

def naive_score(texts,features):
    fl = int(len(features)/2)
    F1 = features[:fl]
    F2 = features[fl:]
    weights = [i for i in range(fl)]*2
    ns = [sum([(f in r)*(fl-w)**1
               for w,f in zip(weights,F1)])-\
          sum([(f in r)*(fl-w)**1 
               for w,f in zip(weights,F2)]) 
                for r in texts]
    return np.array(ns)

def colorize(S,L,c='blue'):
    if type(L)==list:
        for l in L:
            l1 = ' '+l+' '
            S = S.replace(l1,colored(l1,c))
            #re.sub('\s+l',colored(l,c),S)
    elif type(L)==str:
        S = S.replace(L,colored(L,c))
    return S

def history_plot(history,key='acc'):
    vey='val_'+key
    plt.plot(np.array(history.history[key]))
    plt.plot(np.array(history.history[vey]))
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='lower right')
    #plt.yscale(value='log')
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    #plt.yscale(value='log')
    plt.show()