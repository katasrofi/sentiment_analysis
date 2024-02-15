# =============================================================================
# from sklearn.model_selection import train_test_split
# import re
# import nltk
# nltk.download("stopwords")
# from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer
# from numba import jit
# import numpy as np
# 
# def tokenizer_encode(text):
#     chars = sorted(set(text))
#     str_to_int = {ch:i for i, ch in enumerate(chars)}
#     int_to_str = {i:ch for i, ch in enumerate(chars)}
#     encode = lambda x: [str_to_int[c] for c in x]
#     decoder = lambda y: "".join([int_to_str[i] for i in y])
#     
#     return encode, decoder
# 
# def custom_tokenize(text, target):
#     corpus = []
#     y_target = []
#     for i in range(0, len(text)):
#         try:
#            email_type = str(text[i])  # Konversi menjadi string
#            email_type = re.sub('[^a-zA-Z]', ' ', email_type)
#            email_type = email_type.lower()
#            email_type = email_type.split()
#            ps = PorterStemmer()
#            all_stopwords = stopwords.words('english')
#            all_stopwords.remove('not')
#            email_type = [ps.stem(word) for word in email_type if not word in set(all_stopwords)]
#            email_type = ' '.join(email_type)
#            corpus.append(email_type)
#            y_target.append(target[i])
#         except KeyError as e:
#            print("KeyError:", e)
#       
#     return corpus, y_target
# 
# def vectorize(text):
#     Text = text.apply(custom_tokenize)
#     from sklearn.feature_extraction.text import CountVectorizer
#     cv = CountVectorizer(max_features=120000)
#     vectorize_text = cv.fit_transform(Text)
#     return vectorize_text
# 
# def split_train_test_set(text, target):
#     # Menggunakan apply untuk menerapkan custom_tokenize ke setiap baris
#     X = text.apply(custom_tokenize)
#     X = X.apply(vectorize)
#     
#     # Split the data
#     X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.2, random_state=1)
#     return X_train, X_test, y_train, y_test
# 
# 
# =============================================================================
# Torch Tokenize

# =============================================================================
# from torchtext.legacy.data import Field, Example, Dataset
# from torchtext.data.utils import get_tokenizer
# 
# def torch_tokenize(data):
#     token = get_tokenizer("basic_english")
#     TEXT = Field(tokenize=token, lower=True)
#     fields = [("text", TEXT)]
#     data = [Example.fromlist([text], fields) for text in data]
#     dataset = Dataset(data, fields)
#     
#     TEXT.build_vocab(dataset)
#     
#     return dataset
# =============================================================================


def initialize_tokenizer(location):
    from pathlib import Path
    from tokenizers import ByteLevelBPETokenizer
    paths = [str(x) for x in Path(location).glob("**/*.txt")]
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2, special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
        ])
    tokenizer.save_model(".", "custom_Bytetokenizer")
    
    
def tokenizer_preprocessing(vocab, merge):
    from tokenizers.implementations import ByteLevelBPETokenizer
    from tokenizers.processors import BertProcessing
    
    tokenizers = ByteLevelBPETokenizer(
        vocab, merge,
    )
    tokenizers._tokenizer.post_processor = BertProcessing(
        ("</s>", tokenizers.token_to_id("</s>")),
        ("<s>", tokenizers.token_to_id("<s>")),
    )
    tokenizers.enable_truncation(max_length=512)
    return tokenizers
    
from numba import jit
@jit
def preprocessing_text(vocab, merge, text):
    return preprocessing_text(vocab, merge).encode(text)
