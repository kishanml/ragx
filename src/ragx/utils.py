import nltk
import time
import hashlib
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')


def generate_sha256_hash(content : str):
    
    bytes = content.encode()
    hashobj = hashlib.sha256(bytes)
    return hashobj.hexdigest()

def preprocess_text(text : str):
        
    stop_words = set(stopwords.words('english'))
    
    encoded_text = text.lower().encode("ascii", "ignore").decode()
    tokens = word_tokenize(encoded_text)
    filtered_tokens = [word for word in tokens if word not in stop_words]

    return " ".join(filtered_tokens)

class Timer:
    
    def __init__(self, process_name : str):
        self.process_name = process_name
        
    def __enter__(self):
        self.start_time = time.time()
        
    def __exit__(self, exc_type, exc, tb) -> None:
        
        self.end_time = time.time()
        if not exc_type:
            
            print(f'[+] {self.process_name} took - {round(self.end_time-self.start_time,3)}s')
            return  