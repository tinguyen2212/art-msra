import random
from typing import Any, Dict, List, Union

import torch
from torch.utils.data import Dataset

from transformers import DataCollatorForLanguageModeling

from .layout_dataset import LayoutDataset


def get_dataset(args, split: str, quantizer, tokenizer, **kwargs) -> Dataset:
  dataset = LayoutNew(quantizer, tokenizer, split=split, max_len=args.max_len, **kwargs)
  return dataset


class CenterWrapperNew(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.__inner_dataset = dataset

    def __getitem__(self, idx): 
        example = self.__inner_dataset[idx]
        return example
    
    def __len__(self):
        return len(self.__inner_dataset)


class LayoutNew(torch.utils.data.Dataset):
    def __init__(self, quantizer, tokenizer,
                split='train', max_len: int = 32, 
                return_index=False, inf_length=False, 
                with_layout=True,
                split_version='default',
                layout_path = None,
                **kwargs):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.quantizer = quantizer
        poster_datasets = []
        self.split_version = split_version
        self.layout_path = layout_path
        if with_layout:
            if split_version == 'layout':
                print(f"with {split_version} data: {self.layout_path}")
                inner_dataset = LayoutDataset(self.layout_path, split)
            else:
                raise NotImplementedError

            poster_datasets.append(inner_dataset)

        self.inner_dataset = torch.utils.data.ConcatDataset(poster_datasets)
        self.inner_dataset = CenterWrapperNew(self.inner_dataset)
        self.split = split
        self.size = len(self.inner_dataset)
        self.return_index = return_index
        self.inf_length = inf_length
        if self.inf_length:
            self.max_len = 1000000

    def __getitem__(self, idx):
        example = self.inner_dataset[idx]
        json_example = self.quantizer.convert2layout(example)
        max_len = self.max_len
        
        content = self.quantizer.dump2json(json_example)
        
        raw_input_ids = self.tokenizer(content, return_tensors="pt").input_ids[0]
        
        if len(raw_input_ids) > max_len and self.split == 'train':
            start = random.randint(0, len(raw_input_ids) - max_len)
            end = start + max_len
            input_ids = raw_input_ids[start:end]
            
        else:
            input_ids = raw_input_ids
            if not self.inf_length:
                if input_ids.shape[0] > max_len:
                    input_ids = input_ids[:max_len]

        if not self.inf_length:
            if input_ids.shape[0] > self.max_len:
                input_ids = input_ids[:self.max_len]
            elif input_ids.shape[0] < self.max_len:
                padding_1  = torch.zeros((1,), dtype=torch.long) + self.tokenizer.bos_token_id
                padding_2 = torch.zeros((self.max_len - input_ids.shape[0] - 1,), dtype=torch.long) + self.tokenizer.pad_token_id
                input_ids = torch.cat([input_ids, padding_1, padding_2], dim=0)
            assert input_ids.shape[0] == self.max_len

        return_dict = {
            'labels': input_ids,
        }
        if self.return_index:
            return_dict['index'] = idx
        
        return return_dict

    def __len__(self):
        return self.size


def layout_collate_fn(batch):
    input_ids = [item['labels'] for item in batch]
    input_ids = torch.stack(input_ids, dim=0)
    return_dict = {
        'labels': input_ids,
    }
    if 'index' in batch[0]:
        index = [item['index'] for item in batch]
        return_dict['index'] = index
    
    return return_dict


class DataCollatorForLayout(DataCollatorForLanguageModeling):
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        return layout_collate_fn(examples)