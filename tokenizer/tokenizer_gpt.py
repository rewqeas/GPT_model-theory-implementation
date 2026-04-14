import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

######################
#class dataset and dataloader
#########################

class GPTDataset(Dataset):

    def __init__(self,token_ids, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        for i in range(0, len(token_ids) - max_length, stride):
            context = token_ids[i:i + max_length]
            target = token_ids[i + 1:i + max_length + 1]
            self.input_ids.append(torch.tensor(context))
            self.target_ids.append(torch.tensor(target))
    
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self,i):
        return self.input_ids[i], self.target_ids[i]
    

def create_dataloader(token_ids,batch=4, max_length=256, stride=128, shuffle = True, drop_last = True, num_workers = 0,pin_memory = False):


    #create dataset
    dataset = GPTDataset(token_ids, max_length, stride)

    dataloader = DataLoader(
        dataset,
        batch_size = batch,
        shuffle = shuffle,
        drop_last = drop_last,
        num_workers = num_workers,# basically we load the dataset/text into multiple cpu threads for faster data transfer to gpu
        pin_memory=pin_memory #speeds up cpu->gpu tensor transfer successfully
    )
    
    return dataloader

# dataloader = create_dataloader(text)
# data_iter = iter(dataloader)
# first_batch = next(data_iter)
# print(first_batch)