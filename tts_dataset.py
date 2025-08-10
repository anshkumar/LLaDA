import torch
import numpy as np
import json
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from datasets import load_dataset
from typing import List, Dict, Any, Optional, Union
import logging
from transformers import AutoTokenizer


class BatchedRatioDataset(Dataset):
    """
    Dataset that combines text and TTS data in batched ratios
    Based on the original TTS training script
    """
    
    def __init__(
        self, 
        text_dataset, 
        tts_dataset, 
        batch_total: int,
        ratio: float = 0.5,
        tokenizer: Optional[AutoTokenizer] = None
    ):
        self.text_dataset = text_dataset
        self.tts_dataset = tts_dataset
        self.batch_total = batch_total
        self.ratio = int(ratio * 10)  # Convert to integer for easier calculation
        self.tokenizer = tokenizer
        
        # Calculate number of cycles
        num_cycles_text = len(text_dataset) // (batch_total * self.ratio)
        num_cycles_tts = len(tts_dataset) // batch_total
        self.num_cycles = min(num_cycles_text, num_cycles_tts)
        
        # Total length is cycles * (ratio text batches + 1 tts batch) * batch_total
        self.length = self.num_cycles * (self.ratio + 1) * batch_total
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"BatchedRatioDataset: {self.num_cycles} cycles, ratio {self.ratio}:1, total length {self.length}")
    
    def __len__(self):
        return int(self.length)
    
    def __getitem__(self, index):
        # Compute the cycle length in terms of samples
        cycle_length = (self.ratio + 1) * self.batch_total
        cycle = index // cycle_length
        pos_in_cycle = index % cycle_length
        
        if pos_in_cycle < self.ratio * self.batch_total:
            # We are in a text batch
            batch_in_cycle = pos_in_cycle // self.batch_total
            sample_in_batch = pos_in_cycle % self.batch_total
            text_index = cycle * self.ratio * self.batch_total + batch_in_cycle * self.batch_total + sample_in_batch
            
            sample = self.text_dataset[text_index]
            sample['data_type'] = 'text'
            return self._process_sample(sample)
        else:
            # We are in a TTS batch
            sample_in_batch = pos_in_cycle - self.ratio * self.batch_total
            tts_index = cycle * self.batch_total + sample_in_batch
            
            sample = self.tts_dataset[tts_index]
            sample['data_type'] = 'tts'
            return self._process_sample(sample)
    
    def _process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Process a sample to ensure it has the required format"""
        # If input_ids are already present, use them
        if 'input_ids' in sample:
            if isinstance(sample['input_ids'], list):
                sample['input_ids'] = torch.tensor(sample['input_ids'], dtype=torch.long)
        
        # If we have text, tokenize it
        elif 'text' in sample and self.tokenizer is not None:
            tokenized = self.tokenizer(
                sample['text'],
                truncation=True,
                max_length=4096,
                return_tensors='pt'
            )
            sample['input_ids'] = tokenized['input_ids'].squeeze(0)
            if 'attention_mask' in tokenized:
                sample['attention_mask'] = tokenized['attention_mask'].squeeze(0)
        
        # Ensure we have attention mask
        if 'attention_mask' not in sample and 'input_ids' in sample:
            sample['attention_mask'] = torch.ones_like(sample['input_ids'])
        
        # For TTS data, handle prompt length if present
        if sample.get('data_type') == 'tts' and 'prompt_length' not in sample:
            # Try to infer prompt length from special tokens or assume half the sequence
            if 'input_ids' in sample:
                sample['prompt_length'] = len(sample['input_ids']) // 2
        
        return sample


class TTSDataset(Dataset):
    """Dataset specifically for TTS data with audio tokens"""
    
    def __init__(
        self,
        dataset_name_or_path: str,
        tokenizer: AutoTokenizer,
        max_length: int = 4096,
        split: str = "train"
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.logger = logging.getLogger(__name__)
        
        # Load dataset
        if isinstance(dataset_name_or_path, str):
            try:
                # First try load_from_disk for local datasets
                from datasets import load_from_disk
                self.dataset = load_from_disk(dataset_name_or_path)
                self.logger.info(f"Loaded dataset using load_from_disk from {dataset_name_or_path}")
            except Exception as e1:
                try:
                    # Try load_dataset for HuggingFace datasets
                    self.dataset = load_dataset(dataset_name_or_path, split=split)
                    self.logger.info(f"Loaded dataset using load_dataset from {dataset_name_or_path}")
                except Exception as e2:
                    # Finally try as local files
                    self.logger.warning(f"load_from_disk failed: {e1}")
                    self.logger.warning(f"load_dataset failed: {e2}")
                    self.logger.info(f"Trying to load as local files...")
                    self.dataset = self._load_local_dataset(dataset_name_or_path)
        else:
            self.dataset = dataset_name_or_path
        
        self.logger.info(f"Loaded TTS dataset with {len(self.dataset)} examples")
    
    def _load_local_dataset(self, path: str):
        """Load dataset from local files"""
        import os
        if os.path.isfile(path):
            # Single file
            examples = []
            with open(path, 'r') as f:
                if path.endswith('.jsonl'):
                    for line in f:
                        examples.append(json.loads(line.strip()))
                else:
                    examples = json.load(f)
            return examples
        else:
            raise ValueError(f"Cannot load dataset from {path}")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        
        # Process the sample
        if 'input_ids' in sample:
            input_ids = sample['input_ids']
            if isinstance(input_ids, list):
                input_ids = torch.tensor(input_ids, dtype=torch.long)
        elif 'text' in sample:
            # Tokenize text
            tokenized = self.tokenizer(
                sample['text'],
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            input_ids = tokenized['input_ids'].squeeze(0)
        else:
            raise ValueError("Sample must contain either 'input_ids' or 'text'")
        
        # Create attention mask
        attention_mask = torch.ones_like(input_ids)
        
        # Handle prompt length for TTS
        prompt_length = sample.get('prompt_length', len(input_ids) // 2)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'prompt_length': prompt_length,
            'labels': input_ids.clone(),
            'data_type': 'tts'
        }


class TextDataset(Dataset):
    """Dataset specifically for text data"""
    
    def __init__(
        self,
        dataset_name_or_path: str,
        tokenizer: AutoTokenizer,
        max_length: int = 4096,
        split: str = "train"
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.logger = logging.getLogger(__name__)
        
        # Load dataset
        if isinstance(dataset_name_or_path, str):
            try:
                # First try load_from_disk for local datasets
                from datasets import load_from_disk
                self.dataset = load_from_disk(dataset_name_or_path)
                self.logger.info(f"Loaded text dataset using load_from_disk from {dataset_name_or_path}")
            except Exception as e1:
                try:
                    # Try load_dataset for HuggingFace datasets
                    self.dataset = load_dataset(dataset_name_or_path, split=split)
                    self.logger.info(f"Loaded text dataset using load_dataset from {dataset_name_or_path}")
                except Exception as e2:
                    # Finally try as local files
                    self.logger.warning(f"load_from_disk failed: {e1}")
                    self.logger.warning(f"load_dataset failed: {e2}")
                    self.logger.info(f"Trying to load as local files...")
                    self.dataset = self._load_local_dataset(dataset_name_or_path)
        else:
            self.dataset = dataset_name_or_path
        
        self.logger.info(f"Loaded text dataset with {len(self.dataset)} examples")
    
    def _load_local_dataset(self, path: str):
        """Load dataset from local files"""
        import os
        if os.path.isfile(path):
            examples = []
            with open(path, 'r') as f:
                if path.endswith('.jsonl'):
                    for line in f:
                        examples.append(json.loads(line.strip()))
                else:
                    examples = json.load(f)
            return examples
        else:
            raise ValueError(f"Cannot load dataset from {path}")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        
        # Process the sample
        if 'input_ids' in sample:
            input_ids = sample['input_ids']
            if isinstance(input_ids, list):
                input_ids = torch.tensor(input_ids, dtype=torch.long)
        elif 'text' in sample:
            tokenized = self.tokenizer(
                sample['text'],
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            input_ids = tokenized['input_ids'].squeeze(0)
        else:
            raise ValueError("Sample must contain either 'input_ids' or 'text'")
        
        # Create attention mask
        attention_mask = torch.ones_like(input_ids)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone(),
            'data_type': 'text'
        }


class AlternatingDistributedSampler(DistributedSampler):
    """
    Distributed sampler that maintains the alternating pattern of text/TTS data
    Based on the original TTS training script
    """
    
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.shuffle = shuffle
    
    def __iter__(self):
        indices = list(range(len(self.dataset)))
        
        # Don't shuffle to maintain the ratio pattern
        if self.shuffle:
            # If shuffling is enabled, shuffle within each data type block
            # This is more complex and might break the ratio pattern
            pass
        
        # Select indices for this rank
        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)


def tts_data_collator(features: List[Dict[str, Any]], pad_token_id: int = 0) -> Dict[str, torch.Tensor]:
    """
    Data collator for TTS training that handles both text and audio tokens
    Based on the original TTS training script
    """
    # Extract data
    input_ids = [f["input_ids"] for f in features]
    
    # Handle attention masks
    if any("attention_mask" not in f for f in features):
        attention_mask = [torch.ones_like(ids) for ids in input_ids]
    else:
        attention_mask = [f["attention_mask"] for f in features]
    
    # Handle labels
    if any("labels" not in f for f in features):
        labels = [ids.clone() for ids in input_ids]
    else:
        labels = [f["labels"] for f in features]
    
    # Handle prompt lengths for TTS data
    prompt_lengths = []
    for f in features:
        if f.get('data_type') == 'tts' and 'prompt_length' in f:
            prompt_lengths.append(f['prompt_length'])
        else:
            prompt_lengths.append(0)  # Text data doesn't have prompt lengths
    
    # Pad sequences
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(i, dtype=torch.long) if not isinstance(i, torch.Tensor) else i for i in input_ids], 
        batch_first=True, 
        padding_value=pad_token_id
    )
    
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(m, dtype=torch.long) if not isinstance(m, torch.Tensor) else m for m in attention_mask], 
        batch_first=True, 
        padding_value=0
    )
    
    labels = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(l, dtype=torch.long) if not isinstance(l, torch.Tensor) else l for l in labels], 
        batch_first=True, 
        padding_value=-100
    )
    
    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
    
    # Add prompt lengths if we have TTS data
    if any(pl > 0 for pl in prompt_lengths):
        batch["prompt_lengths"] = torch.tensor(prompt_lengths, dtype=torch.long)
    
    # Add data type information
    data_types = [f.get('data_type', 'unknown') for f in features]
    batch["data_types"] = data_types
    
    return batch


def create_tts_datasets(
    config,
    tokenizer: AutoTokenizer
) -> tuple:
    """
    Create text and TTS datasets based on configuration
    
    Returns:
        tuple: (text_dataset, tts_dataset, combined_dataset)
    """
    # Create TTS dataset (always required)
    tts_dataset = TTSDataset(
        config.tts_dataset,
        tokenizer,
        max_length=config.max_length
    )
    
    # Create text dataset only if ratio > 0 and text dataset is different from TTS
    if config.ratio > 0.0 and config.text_dataset != config.tts_dataset:
        text_dataset = TextDataset(
            config.text_dataset,
            tokenizer,
            max_length=config.max_length
        )
        
        # Create combined dataset with ratio
        batch_total = config.batch_size * config.number_processes
        combined_dataset = BatchedRatioDataset(
            text_dataset,
            tts_dataset,
            batch_total,
            ratio=config.ratio,
            tokenizer=tokenizer
        )
    else:
        # TTS only training
        text_dataset = None
        combined_dataset = tts_dataset
    
    return text_dataset, tts_dataset, combined_dataset 