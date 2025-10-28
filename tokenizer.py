from typing import List

class BPETokenizer :
    def __init__(self):
        pass
    
    def train(self, corpus: List[str], vocab_size: int):
        '''Алгоритм BPE'''
        pass
    
    def encode(self, text: str) -> List[int]:
        '''Применить мёрджи -> отобразить в ID'''
        pass
    
    def decode(self, ids: List[int]) -> str:
        '''Обратное преобразование'''
        