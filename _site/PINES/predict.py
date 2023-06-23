import pandas as pd
import numpy as np
import torch.nn.functional as F
from train_finetune import PINES_module
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from transformers import (LongformerConfig,
                          LongformerForSequenceClassification,
                          LongformerTokenizer,
                          get_linear_schedule_with_warmup)

from train_finetune import PINES_module, get_data, PinesDataset

ATTENTION_WINDOW = 512
NUM_LABELS = 2
HF_MODEL = 'allenai/longformer-large-4096'

num_workers = 8
test_batch_sz = 32
devices = 1



tokenizer = LongformerTokenizer.from_pretrained(HF_MODEL)
# _, _, dev_df = get_data()


# dev_dataset = PinesDataset(dev_df.sample(100), tokenizer)
# dev_dataloader = DataLoader(dev_dataset, batch_size=test_batch_sz, shuffle=False, num_workers=num_workers, persistent_workers=True)
data = {"text": ["The patient does not have dvt", "The patient has pain in the calf"], "label": [0, 1]}
dl = DataLoader(PinesDataset(pd.DataFrame.from_dict(data), tokenizer), batch_size=2, shuffle=False, num_workers=1)


model = PINES_module()
best_model = model.load_from_checkpoint("train_steps/longformer-model-epoch=00-val_loss=0.26-v1.ckpt")
trainer = pl.Trainer(accelerator='gpu',
                     devices=devices,
                     precision=16,
                     strategy="ddp",
                     default_root_dir="./train-final-1"
)

best_model.eval()
res = trainer.predict(best_model, dl)
# dev_preds = torch.load('dev_preds_new.pt')
all_probabilities = []
for outputs in res:
    probabilities = F.softmax(outputs, dim=-1)
    # Append probabilities to list
    all_probabilities.extend(probabilities[:, 1].tolist())
print(all_probabilities)