import re
import json
import argparse
from typing import List

import torch
from transformers import AutoTokenizer, set_seed
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList, \
    STOPPING_CRITERIA_INPUTS_DOCSTRING, add_start_docstrings

from models.modeling_layout import LayoutModel, LayoutModelConfig
from training.datasets.quantizer import get_quantizer


class StopAtSpecificTokenCriteria(StoppingCriteria):
    def __init__(self, token_id_list: List[int] = None):
        self.token_id_list = token_id_list
    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return input_ids[0][-1].detach().cpu().numpy() in self.token_id_list


# build model and tokenizer
def buildmodel(device='cuda:0',**kwargs):
    # seed / input model / resume
    resume = kwargs.get('resume', None)
    seed = kwargs.get('seed', None)
    input_model = kwargs.get('input_model', None)
    quantizer_version = kwargs.get('quantizer_version', 'v4')

    set_seed(seed)
    old_tokenizer = AutoTokenizer.from_pretrained(input_model, trust_remote_code=True)
    old_vocab_size = len(old_tokenizer)
    print(f"Old vocab size: {old_vocab_size}")
    
    tokenizer = AutoTokenizer.from_pretrained(resume, trust_remote_code=True)

    new_vocab_size = len(tokenizer)
    print(f"New vocab size: {new_vocab_size}")
    quantizer = get_quantizer(quantizer_version, 
                    simplify_json = True,
                    width = kwargs['width'],
                    height = kwargs['height']
                    )
    quantizer.setup_tokenizer(tokenizer)    
    print(f"latest tokenzier size: {len(tokenizer)}")

    model_args = LayoutModelConfig(
        old_vocab_size = old_vocab_size,
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        ignore_ids=tokenizer.convert_tokens_to_ids(quantizer.ignore_tokens),
    )

    model_args.opt_version = input_model
    model_args.freeze_lm = False
    model_args.load_in_4bit = kwargs.get('load_in_4bit', False)
 
    print(f"Resuming from checkpoint {resume}, Waiting to ready")
    model = LayoutModel.from_pretrained(resume, config=model_args).to(device)

    return model, quantizer, tokenizer


def preprocess_Input(intention: str):

    intention = intention.replace('\n', '').replace('\r', '').replace('\\', '')
    intention = re.sub(r'\.\s*', '. ', intention)

    return intention


# build data
def FormulateInput(intention: str):
    '''
    Formulate user input string to Dict Object
    '''
    resdict = {}
    resdict["wholecaption"] = intention
    resdict["layout"] = []

    return resdict


@torch.no_grad()    
def evaluate_v1(inputs, model, quantizer, tokenizer, width, height, device, do_sample=False, temperature=1.0, top_p=1.0, top_k=50):
    json_example = inputs
    input_intention = '{"wholecaption":"' + json_example["wholecaption"] + '","layout":[{"layer":'
    print("input_intention:\n", input_intention)

    inputs = tokenizer(
            input_intention, return_tensors="pt"
        ).to(device)

    stopping_criteria = StoppingCriteriaList()
    stopping_criteria.append(StopAtSpecificTokenCriteria(token_id_list=[128000]))

    outputs = model.lm.generate(**inputs, use_cache=True, max_length=8000, stopping_criteria=stopping_criteria, do_sample=do_sample, temperature=temperature, top_p=top_p, top_k=top_k)
    inputs_length = inputs['input_ids'].shape[1] 
    outputs = outputs[:, inputs_length:]

    outputs_word = tokenizer.batch_decode(outputs)[0]
    split_word = outputs_word.split('}]}')[0]+"}]}"
    split_word = '{"wholecaption":"' + json_example["wholecaption"].replace('\n', '\\n').replace('"', '\\"') + '","layout":[{"layer":' + split_word

    map_dict = quantizer.construct_map_dict()
    for key ,value in map_dict.items():
        split_word = split_word.replace(key, value)

    try:
        pred_json_example = json.loads(split_word)
        for layer in pred_json_example["layout"]:
            layer['x'] = round(int(width)*layer['x'])
            layer['y'] = round(int(height)*layer['y'])
            layer['width'] = round(int(width)*layer['width'])
            layer['height'] = round(int(height)*layer['height'])
    except Exception as e:
        print(e)
        pred_json_example = None
    return pred_json_example


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--inference_caption', type=str, help='User input whole caption')
    parser.add_argument('--save_path', type=str, help='Path to save data')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--width', type=int, default=1024, help='Width of the layout')
    parser.add_argument('--height', type=int, default=1024, help='Height of the layout')
    parser.add_argument('--input_model', type=str, help='Path to input base model')
    parser.add_argument('--resume', type=str, help='Path to test model checlpoint')
    parser.add_argument('--do_sample', type=bool, default=False)
    parser.add_argument('--temperature', type=float, default=0.5)

    args = parser.parse_args()

    inference_caption = args.inference_caption
    save_path = args.save_path
    device = args.device
    width = args.width
    height = args.height
    input_model = args.input_model
    resume = args.resume
    do_sample = args.do_sample
    temperature = args.temperature

    params_dict = {
        "input_model": input_model,
        "resume": resume,
        "seed": 0, 
        "quantizer_version": 'v4',
        "width": width,
        "height": height,
    }

    # Init model
    model, quantizer, tokenizer = buildmodel(device=device, **params_dict)
    model = model.to(device)
    model = model.bfloat16()
    model.eval()

    intention = preprocess_Input(inference_caption)
    rawdata = FormulateInput(intention)
    preddata = evaluate_v1(rawdata, model, quantizer, tokenizer, width, height, device, do_sample=do_sample, temperature=temperature)
    max_try_time = 3
    while preddata is None and max_try_time > 0:
        preddata = evaluate_v1(rawdata, model, quantizer, tokenizer, width, height, device, do_sample=do_sample, temperature=temperature)
        max_try_time -= 1

    print("output : ", preddata)

    with open(save_path,'w') as file:
        json.dump(preddata, file, indent=4)
