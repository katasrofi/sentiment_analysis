import cython
from sklearn.model_selection cimport train_test_split
cimport re
cimport nltk
nltk.download("stopwords")
from nltk.corpus cimport stopwords
from nltk.stem.porter cimport PorterStemmer
from numba cimport jit
cimport pandas as pd
cimport numpy as np

def custom_tokenize_optim(str text):
    # Declare varibale
    cdef str text_str, text_clean, text_lower, ps, all_stopwords, X_token
    cdef list X = []
    
    # Loop 
    for i in range(len(text)):
        text_str = str(text)[i]
        
        # Clean The data
        text_clean = re.sub('[^a-zA-Z]', '', text_str)
        
        # Preprocessing the data
        text_lower = text_clean.lower()
        corpus = text_lower.split()
        
        # Tokenizer
        ps = PorterStemmer()
        all_stopwords = stopwords.words("English")
        all_stopwords.remove("not")
        X_token = [ps.stem(word) for word in corpus if not word in set(all_stopwords)]
        X_token = " ".join(X)
        X.append(X_token)
        
    return X
    
    
cimport sklearn.model_selection as skms

def split_train_test_set(np.ndarray[np.float64_t, ndim=2] text, np.ndarray[np.float64_t, ndim=1] target):
    X = custom_tokenize_optim(text)
    cdef np.ndarray[np.float64_t, ndim=2] X_train, X_test
    cdef np.ndarray[np.float64_t, ndim=1] y_train, y_test
    # Split the data
    X_train, y_train, X_test, y_test = skms.train_test_split(X, 
                                                        target, 
                                                        test_size=0.2, 
                                                        random_state=1)
    return X_train, y_train, X_test, y_test
