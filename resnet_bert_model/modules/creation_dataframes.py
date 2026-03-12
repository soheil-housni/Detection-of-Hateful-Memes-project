import pandas as pd
import json

#This function enables to transform the jsonl dataset into a pandas dataframe 
def creation_dataframe(path : str) -> pd.DataFrame:
    data=[]
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    df=pd.DataFrame(data).drop(columns="id")
    return df