import torch
from transformers import AutoTokenizer

class CollateFunction():
    def __init__(self,tokenizer:AutoTokenizer):
        self.tokenizer=tokenizer
    
    """
    Custom collate function for the creation of the Dataloader respecting Pytorch standards
    """
    def collate_fn(self,batch:list[dict]) -> dict:
        images=[b["images"] for b in batch]
        texts=[b["texts"] for b in batch]
        labels=[b["labels"] for b in batch]
        texts=self.tokenizer(texts,return_tensors="pt",max_length=128,padding="max_length",truncation=True)
        images=torch.stack(images,dim=0)
        labels=torch.tensor(labels,dtype=torch.long)
        inputs={"images":images,"labels":labels}
        inputs.update(texts)
        return inputs