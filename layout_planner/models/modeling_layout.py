import os
import torch
from typing import Optional, List
from transformers import PreTrainedModel, PretrainedConfig, AutoModel, AutoModelForCausalLM, OPTForCausalLM, BitsAndBytesConfig


def kmp_preprocess(pattern):
    pattern_len = len(pattern)
    prefix_suffix = [0] * pattern_len
    j = 0

    for i in range(1, pattern_len):
        while j > 0 and pattern[i] != pattern[j]:
            j = prefix_suffix[j - 1]

        if pattern[i] == pattern[j]:
            j += 1

        prefix_suffix[i] = j

    return prefix_suffix


def kmp_search(text, pattern):
    text_len = len(text)
    pattern_len = len(pattern)
    prefix_suffix = kmp_preprocess(pattern)
    matches = []

    j = 0
    for i in range(text_len):
        while j > 0 and text[i] != pattern[j]:
            j = prefix_suffix[j - 1]

        if text[i] == pattern[j]:
            j += 1

        if j == pattern_len:
            matches.append(i - j + 1)
            j = prefix_suffix[j - 1]

    return matches


class ModelWrapper:
    def __init__(self, model):
        self.model = model

    def __getattr__(self, name):
        return getattr(self.model, name)

    @torch.no_grad()
    def __call__(self, pixel_values):
        return self.model(pixel_values)
  
    def eval(self):
        pass

    def train(self):
        pass

    def parameters(self):
        return self.model.parameters()


class LayoutModelConfig(PretrainedConfig):
    def __init__(
        self,
        old_vocab_size: int = 32000,
        vocab_size: int = 32000,
        pad_token_id: int = 2,
        freeze_lm: bool = True,
        opt_version: str = 'facebook/opt-6.7b',
        hidden_size: int = -1,
        load_in_4bit: Optional[bool] = False,
        ignore_ids: List[int] = [],
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert old_vocab_size > 0, 'old_vocab_size must be positive'
        assert vocab_size > 0, 'vocab_size must be positive'

        self.old_vocab_size = old_vocab_size
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.freeze_lm = freeze_lm
        self.opt_version = opt_version
        self.hidden_size = hidden_size
        self.load_in_4bit = load_in_4bit
        self.ignore_ids = ignore_ids


class LayoutModel(PreTrainedModel):
    config_class = LayoutModelConfig
    supports_gradient_checkpointing = True
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.lm.gradient_checkpointing_enable()
    
    def __init__(self, config: LayoutModelConfig):
        super().__init__(config)
        self.pad_token_id = config.pad_token_id
    
        self.args = config
      
        opt_version = config.opt_version
      
        print(f"Using {opt_version} for the language model.")
    
        if config.load_in_4bit:
            print("\n would load_in_4bit")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=config.load_in_4bit
            )
            # This means: fit the entire model on the GPU:0
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            device_map = {"": local_rank}
            torch_dtype = torch.bfloat16
        else:
            print("\n wouldn't load_in_4bit")
            device_map = None
            quantization_config = None
            torch_dtype = None
    
        self.lm = AutoModelForCausalLM.from_pretrained(
            opt_version,
            quantization_config=quantization_config,
            device_map=device_map,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
        )
        self.config.hidden_size = self.lm.config.hidden_size
        self.opt_version = opt_version
    
        if self.args.freeze_lm:
            self.lm.eval()
            print("Freezing the LM.")
            for param in self.lm.parameters():
                param.requires_grad = False
        else:
            print("\n no freeze lm, so to train lm")
            self.lm.train()
            self.lm.config.gradient_checkpointing = True
    
        print('resize token embeddings to match the tokenizer', config.vocab_size)
        self.lm.resize_token_embeddings(config.vocab_size)
        self.input_embeddings = self.lm.get_input_embeddings()
    
    def train(self, mode=True):
        super().train(mode=mode)
        # Overwrite train() to ensure frozen models remain frozen.
        if self.args.freeze_lm:
            self.lm.eval()
    
    def forward(
        self,
        labels: torch.LongTensor,
    ):
        batch_size = labels.shape[0]
        full_labels = labels.detach().clone()
      
        input_embs = self.input_embeddings(labels)  # (N, T, D)
        input_embs_norm = ((input_embs ** 2).sum(dim=-1) ** 0.5).mean()
      
        for ignore_id in self.config.ignore_ids:
            full_labels[full_labels == ignore_id] = -100
    
        pad_idx = []
        # -100 is the ignore index for cross entropy loss. https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        for label in full_labels:
            for k, token in enumerate(label):
                # Mask out pad tokens if they exist.
                if token in [self.pad_token_id]:
                    label[k:] = -100
                    pad_idx.append(k)
                    break
                if k == len(label) - 1:  # No padding found.
                    pad_idx.append(k + 1)
        assert len(pad_idx) == batch_size, (len(pad_idx), batch_size)
      
        output = self.lm( inputs_embeds=input_embs,
                          labels=full_labels,
                          output_hidden_states=True)
        
        return output, full_labels, input_embs_norm