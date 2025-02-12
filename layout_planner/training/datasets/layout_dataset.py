import json
import torch


class LayoutDataset(torch.utils.data.Dataset):
    def __init__(self, layout_data_path, split_name):
        self.split_name = split_name
        self.layout_data_path = layout_data_path
        with open(self.layout_data_path, 'r') as file:
            self.json_datas = json.load(file)
        
        self.ids = [x for x in range(len(self.json_datas))]
        if self.split_name == 'train':
            self.ids = self.ids[:-200]
            self.json_datas = self.json_datas[:-200]           
        elif self.split_name == 'val':
            self.ids = self.ids[-200:-100]
            self.json_datas = self.json_datas[-200:-100]
        elif self.split_name == 'test':
            self.ids = self.ids[-100:]
            self.json_datas = self.json_datas[-100:]
        else:
            raise NotImplementedError

    def __getitem__(self, idx):
        return self.json_datas[idx]['train']

    def __len__(self):
        return len(self.json_datas)