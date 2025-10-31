from typing import List, Dict, Tuple, Optional
import re
from collections import defaultdict, Counter
import os 
import json

class BPETokenizer :
    def __init__(self, special_tokens: Optional[Dict[str, int]] = None):
        self.special_tokens = special_tokens or {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
        self.merges = [] #список мёрджей
        self.vocab = {} #token -> id
        self.inv_vocab = {} #id-> token
        self._is_trained = False
        
    def _get_stats(self, words: List[List[str]]) -> Dict[Tuple[str, str], int]:
        '''count frequency of adjacent symbol pairs.'''
        pairs = defaultdict(int)
        for word in words:
            for i in range(len(word) - 1):
                pairs[(word[i], word[i+1])] += 1
        return pairs
    
    def _merge_vocab(self, pair: Tuple[str, str], words: List[List[str]]) -> List[List[str]]:
        '''merge all occerrences of a pair in the vocab'''
        new_words = []
        bigram = ''.join(pair)
        for word in words:
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i+1]) == pair:
                    new_word.append(bigram)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_words.append(new_word)
        return new_words
        
    def train(self, file_path: str, vocab_size:int, min_frequency:int = 2, max_lines:int = None):
        '''Алгоритм BPE'''
        print("Started Training")
        with open(file_path, 'r', encoding='utf-8') as f:
            if max_lines:
                lines = [f.readline() for _ in range(max_lines)]
            
            else:
                lines = f.readlines()
                
        #split into words and paste end-of-token 
        word_freqs = defaultdict(int)
        for line in lines:
            words = line.strip().split()
            for word in words:
                word_freqs[word + '</w>'] += 1
                
        #initialize vocab as characters
        vocab = []
        for word, freq in word_freqs.items():
            vocab.extend([[char for char in word]] * freq)
            
        print(f"Initial vocab size (chars): {len(set(ch for word in vocab for ch in word))}")
    
        #Learn BPE merges
        num_merges = vocab_size - len(self.special_tokens) - len(set(ch for word in vocab for ch in word))
        num_merges = max(0, num_merges)
        
        print(f"Learning {num_merges} merge opearations...")
        for i in range (num_merges):
            pairs = self._get_stats(vocab)
            if not pairs:
                break
            
            best_pair = max(pairs, key=pairs.get)
            if pairs[best_pair] < min_frequency:
                break
            
            vocab = self._merge_vocab(best_pair, vocab)
            self.merges.append(best_pair)
            
            if (i + 1) % 100 == 0:
                print(f"    {i+1}/{num_merges} merges done")
                
        #final vocab building
        all_tokens = set(self.special_tokens.keys())
        for word in vocab:
            all_tokens.update(word)
            
        sorted_tokens = sorted(all_tokens)
        next_id = len(self.special_tokens)
        
        for token in sorted_tokens:
            if token not in self.special_tokens:
                self.vocab[token] = next_id
                self.inv_vocab[next_id] = token
                next_id += 1
        self._is_trained = True
        print(f"Final vocab size:{len(self.vocab)}")
    
    def tokenize(self, text: str) -> List[str]:
        '''Tokenize text into subword units(not IDs)'''
        if not self._is_trained:
            raise RuntimeError("Tokenizer not trained yet!")
        
        words = text.strip().split()
        tokenized_words = []
        for word in words:
            word = word + '</w>'
            #start with characters
            tokens = list(word)
            
            #applying all learned merges
            for pair in self.merges:
                new_tokens = []
                i = 0
                while i < len(tokens):
                    if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == pair:
                        new_tokens.append(''.join(pair))
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                tokens = new_tokens
            tokenized_words.extend(tokens)
        return tokenized_words
        
    def save(self, path: str):
        '''Saves tokenizer state to json'''
        state = {
            "special_tokens": self.special_tokens,
            "merges": self.merges,
            "vocab": self.vocab
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    
    def encode(self, text: str) -> List[int]:
        '''Применить мёрджи -> отобразить в ID'''
        tokens = self.tokenize(text)
        ids = []
        for token in tokens:
            if token in self.vocab:
                ids.append(self.vocab[token])
            else:
                ids.append(self.special_tokens["<UNK>"])
        return ids
    
    def decode(self, ids: List[int]) -> str:
        '''Обратное преобразование'''
        tokens = []
        for id_ in ids:
            if id_ in self.inv_vocab:
                tokens.append(self.inv_vocab[id_])
            elif id_ in self.special_tokens.values():
                #skip special tokens in output
                continue
            else:
                tokens.append("<UNK>")
                
        #join and remove </w> markers
        text = ''.join(tokens)
        text = text.replace('</w>', ' ')
        #clean extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    @classmethod
    def load_tokenizer(cls, path: str):
        """Load tokenizer from file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        tokenizer = BPETokenizer()
        tokenizer.vocab = {k: v for k, v in data['vocab'].items()}
        tokenizer.merges = [tuple(m) for m in data['merges']]
        tokenizer.special_tokens = data['special_tokens']
        # Build inverse vocab
        tokenizer.inv_vocab = {v: k for k, v in tokenizer.vocab.items()}
        tokenizer._is_trained = True
        return tokenizer
    

    @property
    def vocab_size(self):
        return max(self.vocab.values()) + 1