import json
import copy
import numpy as np
from functools import lru_cache
from collections import OrderedDict


class BaseQuantizer:
    @property
    def ignore_tokens(self):
        return []

    def __init__(self, simplify_json=False, **kwargs):
        self.simplify_json=simplify_json
        self.io_ignore_replace_tokens = ['<split-text>']

    def dump2json(self, json_example):
        if self.simplify_json:
            content = json.dumps(json_example, separators=(',',':'))
            for token in self.additional_special_tokens:
                content = content.replace(f'"{token}"', token)
        else:
            content = json.dumps(json_example)
        return content

    def load_json(self, content):
        replace_tokens = set(self.additional_special_tokens) - set(self.io_ignore_replace_tokens) # sirui change 
        if self.simplify_json:
            for token in replace_tokens:
                content = content.replace(token, f'"{token}"')
        return json.loads(content)


specs={
    "width":"size",
    "height":"size",
    "x":"pos", # center x
    "y":"pos", # center y
    "color":"color",
    "font":"font"
}


min_max_bins = {
    'size': (0,1,256),
    'pos': (0,1,256),
    'color': (0,137,138),
    'font': (0,511,512)
}


class QuantizerV4(BaseQuantizer):
    def __init__(self, quant=True, **kwargs):
        super().__init__(**kwargs)
        self.min = min
        self.max = max
        self.quant = quant
        self.text_split_token = '<split-text>'
        self.set_min_max_bins(min_max_bins)
        self.width = kwargs.get('width', 1024)
        self.height = kwargs.get('height', 1024)
        self.width = int(self.width)
        self.height = int(self.height)

    def set_min_max_bins(self, min_max_bins):
        min_max_bins = copy.deepcopy(min_max_bins)
        # adjust the bins to plus one
        for type_name, (min_val, max_val, n_bins) in min_max_bins.items():
            assert n_bins % 2 == 0
            min_max_bins[type_name] = (min_val, max_val, n_bins+1)
        self.min_max_bins = min_max_bins

    def setup_tokenizer(self, tokenizer):
        additional_special_tokens = [self.text_split_token]
        rest_types = [key for key in self.min_max_bins.keys()]
        for type_name in rest_types:
            min_val, max_val,n_bins = self.min_max_bins[type_name]
            additional_special_tokens += [f'<{type_name}-{i}>' for i in range(n_bins)]

        print('additional_special_tokens', additional_special_tokens)

        tokenizer.add_special_tokens({'additional_special_tokens': additional_special_tokens})
        self.additional_special_tokens = set(additional_special_tokens)
        return tokenizer


    @lru_cache(maxsize=128)
    def get_bins(self, real_type):
        min_val, max_val, n_bins = self.min_max_bins[real_type]
        return min_val, max_val, np.linspace(min_val, max_val, n_bins)

    def quantize(self, x, type):
        if not self.quant:
            return x
        """Quantize a float array x into n_bins discrete values."""
        real_type = specs[type]
        min_val, max_val, bins = self.get_bins(real_type)
        x = np.clip(float(x), min_val, max_val)
        val = np.digitize(x, bins) - 1
        n_bins = len(bins)
        assert val >= 0 and val < n_bins
        return f'<{real_type}-{val}>'
    
    def dequantize(self, x):
        # <pos-1>->1
        val = x.split('-')[1].strip('>')
        # <pos-1>->pos
        real_type = x.split('-')[0][1:]
        min_val, max_val, bins = self.get_bins(real_type)
        return bins[int(val)]

    def construct_map_dict(self):
        map_dict = {}
        for i in range(self.min_max_bins['size'][2]):
            name = "<size-%d>" % i
            value = self.dequantize(name)
            map_dict[name] = str(value)
        for i in range(self.min_max_bins['pos'][2]):
            name = "<pos-%d>" % i
            value = self.dequantize(name)
            map_dict[name] = str(value)
        return map_dict
    
    def postprocess_colorandfont(self, json_example):
        import re
        json_example = re.sub(r'(<font-\d+>)', r'"\1"', json_example)
        json_example = re.sub(r'(<color-\d+>)', r'"\1"', json_example)
        return json_example

    def convert2layout(self, example):
        new_example = OrderedDict()
        new_example['wholecaption'] = example['wholecaption']
        new_layout = []
        for meta_layer in example['layout']:
            new_layout.append({
            "layer": meta_layer["layer"],
            "x": self.quantize(meta_layer["x"]/self.width, 'x'),
            "y": self.quantize(meta_layer["y"]/self.height, 'y'),
            "width": self.quantize(meta_layer["width"]/self.width, 'width'),
            "height": self.quantize(meta_layer["height"]/self.height, 'height')
        })
        new_example['layout'] = new_layout
        return new_example


def get_quantizer(version='v1', **kwargs):
    if version == 'v4':
        quantizer = QuantizerV4(**kwargs)
    else:
        raise NotImplementedError

    return quantizer
    
