import numpy as np
from functools import wraps
from typing import overload
import datetime as dt
import pandas as pd
import seaborn as sns
import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchmetrics
from torchmetrics import Accuracy
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import (LongformerConfig,
                          LongformerForSequenceClassification,
                          LongformerTokenizer,
                          get_linear_schedule_with_warmup)

from pynvml import *

torch.manual_seed(42)
seed_everything(42)


ATTENTION_WINDOW = 512
NUM_LABELS = 2
HF_MODEL = 'allenai/longformer-large-4096'
tokenizer = LongformerTokenizer.from_pretrained(HF_MODEL)


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle=handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()


def logg(f):
    @wraps(f)
    def wrapper(dataf, *args, **kwargs):
        tic = dt.datetime.now()
        result = f(dataf, *args, **kwargs)
        toc = dt.datetime.now()
        print(f"{f.__name__} took={toc - tic} shape={result.shape}")
        return result
    return wrapper


@logg
def start_pipeline(dataf):
    return dataf.copy()


@logg
def remove_duplicated(dataf):
    dataf = dataf[dataf.note_text!="""***DUPLICATED***"""]
    dataf = dataf.drop_duplicates(subset = ["doc_id"])
    return dataf


@logg
def get_relevant_columns(dataf, cols=["patient_id", "note_text", "text_date"]):
    dataf = dataf[cols]
    dataf = dataf.rename(columns = {"note_text" : "text"})
    return dataf

@logg
def update_dtype(dataf, dtype: dict) -> pd.DataFrame:
    for col, col_type in dtype.items():
        if col_type == "datetime":
            dataf[col] = pd.to_datetime(dataf[col])
        else:
            raise NotImplementedError(f'Converting to "{col_type}" is not implemented')
    return dataf

@logg
def map_column_name(dataf, rename_cols: dict):
    dataf = dataf.rename(columns=rename_cols)
    return dataf


@logg 
def create_labels(dataf):
    dataf["label"] = (dataf["text_date"] >= dataf["event_date"]).astype(int)
    dataf = dataf.rename(columns={"text_date": "date"})
    return dataf[["patient_id", "text", "date", "label"]]


def get_data():
    # Getting events

    events = pd.read_parquet("../In/prepped_core_7_21_2022.parquet",
                            columns=["MRN", "CANCER_VTE_DATE", "OLD_VTE_DATE"]
                            )\
    .pipe(start_pipeline)\
    .pipe(update_dtype, dtype={"CANCER_VTE_DATE": "datetime", 
                                "OLD_VTE_DATE": "datetime"})\
    .pipe(map_column_name, rename_cols={"MRN": "patient_id",
                                    "CANCER_VTE_DATE": "event_date"})


    # Getting texts

    notes_train = pd.read_csv("../In/aggregated_train.csv", low_memory=False)
    notes_val = pd.read_csv("../In/aggregated_val.csv", low_memory=False)
    notes_dev = pd.read_csv("../In/aggregated_dev.csv", low_memory=False)



    print("*** Getting Train Data ***")
    notes_train_inter = notes_train.pipe(start_pipeline).pipe(remove_duplicated).pipe(get_relevant_columns).pipe(update_dtype, {"text_date": "datetime"})
    notes_train_df = pd.merge(events, notes_train_inter)
    train_df = notes_train_df.pipe(start_pipeline).pipe(create_labels)

    print("*** Getting Val Data ***")
    notes_val_inter = notes_val.pipe(start_pipeline).pipe(remove_duplicated).pipe(get_relevant_columns).pipe(update_dtype, {"text_date": "datetime"})
    notes_val_df = pd.merge(events, notes_val_inter)
    val_df = notes_val_df.pipe(start_pipeline).pipe(create_labels)

    print("*** Getting Dev Data ***")
    notes_dev_inter = notes_dev.pipe(start_pipeline).pipe(remove_duplicated).pipe(get_relevant_columns).pipe(update_dtype, {"text_date": "datetime"})
    notes_dev_df = pd.merge(events, notes_dev_inter)
    dev_df = notes_dev_df.pipe(start_pipeline).pipe(create_labels)

    return train_df, val_df, dev_df


class PinesDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data: pd.DataFrame, 
                 tokenizer: LongformerTokenizer,
                 max_token_length: float=4096):
        self.data = data
        self.max_token_length = max_token_length
        self.new_toks = ["dvt", "pe", "vte", "thrombosis", "thromboses", "thrombus", "thrombi", "thrombotic", "clot", "embolus", "emboli", "thrombophlebitis"]
        self.expanded_tokens = self.__expand_tokens()
        tokenizer.add_tokens(self.expanded_tokens)
        self.global_mask_ids = torch.tensor(tokenizer.encode_plus(self.expanded_tokens)["input_ids"])
    
    def __expand_tokens(self):
        new_toks_app = []
        for x in self.new_toks:
            new_toks_app.append(x)
            new_toks_app.append(" " + x)
            new_toks_app.append(x.capitalize())
            new_toks_app.append(" " + x.capitalize())   
            new_toks_app.append(x.upper())
            new_toks_app.append(" " + x.upper())
        return(new_toks_app)

    def __len__(self):
        return len(self.data)
    
    def classes(self):
        return self.data.label.to_list()

    def __getitem__(self, index):
        subset_data = self.data.iloc[index]
        
        text = subset_data["text"]
        label = subset_data.label
        
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_token_length,
            truncation=True,
            return_attention_mask=True,
            padding="max_length",
            return_tensors='pt'
        )
        
        input_ids=encoding["input_ids"].flatten()
        attention_mask=encoding["attention_mask"].flatten()
        global_mask = torch.isin(input_ids, self.global_mask_ids).type(torch.int64)
        global_mask[0] = 1 # cls token to 1
        return dict(
            # report_text=text,
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_mask,
            labels=label
        )


class PINES_module(pl.LightningModule):
    
    def __init__(self, 
                 num_labels=2,
                 learning_rate: float = 1e-6,
                 adam_epsilon: float = 1e-8,
                 warmup_steps: int = 1000,
                 weight_decay: float = 0.01,
                 **kwargs):
        super().__init__()
        
        # PL attributes
        self.save_hyperparameters()
        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()
        self.config = LongformerConfig.from_pretrained(HF_MODEL, attention_window = ATTENTION_WINDOW, attention_mode='sliding_chunks')
        self.model = LongformerForSequenceClassification.from_pretrained(HF_MODEL, config = self.config)
        # self.model.train()
        self.model.resize_token_embeddings(len(tokenizer))

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        train_loss, logits = outputs[:2]
        preds = torch.argmax(logits, axis=1)  
        labels = batch["labels"]
        self.train_acc(preds,labels)
        self.log('train_acc', self.train_acc, on_epoch=True)
        loss = outputs[0]
        return loss
        
    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        val_loss, logits = outputs[:2]
        preds = torch.argmax(logits, axis=1)  
        labels = batch["labels"]
        self.valid_acc(preds,labels)
        self.log('valid_acc', self.valid_acc, on_epoch=True)
        return {"loss": val_loss, "preds": preds, "labels": batch["labels"]}
    
    def validation_epoch_end(self, outputs):
        preds = torch.cat([x["preds"] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy()
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("val_loss", loss, prog_bar=True)
        # self.log_dict(self.valid_acc.compute(predictions=preds, references=labels), prog_bar=True)  
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        outputs = self(**batch)
        return outputs.logits
    
    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        
        return [optimizer], [scheduler]
    

if __name__ == "__main__":

    model = PINES_module()

    dry_run = False
    num_workers = 8
    train_batch_sz = 2
    test_batch_sz = 32
    devices = 4
    checkpoint = True
    epochs = 15

    if dry_run:
        num_workers = 2
        train_batch_sz = 2
        test_batch_sz = 32
        devices = 4
        checkpoint = True
        epochs = 2

    train_df, val_df, dev_df = get_data()

    if dry_run:
        train_dataset = PinesDataset(train_df[:64], tokenizer)
        val_dataset = PinesDataset(val_df.sample(100), tokenizer)
        dev_dataset = PinesDataset(dev_df.sample(100), tokenizer)
    else:
        train_dataset = PinesDataset(train_df, tokenizer)
        val_dataset = PinesDataset(val_df, tokenizer)
        dev_dataset = PinesDataset(dev_df, tokenizer)

    
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_sz, shuffle=True, num_workers=num_workers, persistent_workers=True)
    val_dataloader = DataLoader(val_dataset, batch_size=train_batch_sz, shuffle=False, num_workers=num_workers, persistent_workers=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=test_batch_sz, shuffle=False, num_workers=num_workers, persistent_workers=True)
    
    # Create a ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='train_steps',
        filename='longformer-model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
)
    # training
    trainer = pl.Trainer(enable_checkpointing=checkpoint,
                         accelerator='gpu',
                         devices=devices,
                         max_epochs=epochs,
                         precision=16,
                         strategy="ddp_spawn",
                         default_root_dir="./train-final-1",
                         callbacks=[checkpoint_callback],
                         val_check_interval=5000
                         )
    
    trainer.fit(model, train_dataloader, val_dataloader)