from transformers import AutoProcessor
import torch
"""
Custom collate function for the dataloader used for the forward of the CLIP model
that process images and texts of the batch with the processor using standards of CLIP.
"""
class FLAVACollateFunction():
    def __init__(self,processor:AutoProcessor):
        self.processor=processor
 
    def collate_fn(self,batch:list[dict]) -> dict:
        images=[b["images"] for b in batch]
        texts=[b["texts"] for b in batch]
        labels=[b["labels"] for b in batch]
        inputs=self.processor(images,texts,return_tensors="pt",padding="max_length",truncation=True, max_length=128)
        inputs["labels"]=torch.tensor(labels,dtype=torch.float32)
        return inputs