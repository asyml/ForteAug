# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 00:19:30 2020
@author: Jiang Yuxin
"""

import torch
from torch import nn
from transformers import (
    BertForSequenceClassification,
    AlbertForSequenceClassification,
    XLNetForSequenceClassification,
    RobertaForSequenceClassification,
    AutoTokenizer,
)
from data_sst import ClassificationDataProcessor
from torch.utils.data import DataLoader
from transformers.optimization import AdamW
from utils import train


class AlbertModel(nn.Module):
    def __init__(self, requires_grad=True):
        super(AlbertModel, self).__init__()
        self.albert = AlbertForSequenceClassification.from_pretrained(
            "albert-xxlarge-v2", num_labels=2
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "albert-xxlarge-v2", do_lower_case=True
        )
        self.requires_grad = requires_grad
        self.device = torch.device("cuda")
        for param in self.albert.parameters():
            param.requires_grad = True  # Each parameter requires gradient

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments, labels):
        loss, logits = self.albert(
            input_ids=batch_seqs,
            attention_mask=batch_seq_masks,
            token_type_ids=batch_seq_segments,
            labels=labels,
        )[:2]
        probabilities = nn.functional.softmax(logits, dim=-1)
        return loss, logits, probabilities


class BertModel(nn.Module):
    def __init__(self, requires_grad=True):
        super(BertModel, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(
            "textattack/bert-base-uncased-SST-2", num_labels=2
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "textattack/bert-base-uncased-SST-2", do_lower_case=True
        )
        self.requires_grad = requires_grad
        self.device = torch.device("cuda")
        for param in self.bert.parameters():
            param.requires_grad = (
                requires_grad  # Each parameter requires gradient
            )

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments, labels):
        loss, logits = self.bert(
            input_ids=batch_seqs,
            attention_mask=batch_seq_masks,
            token_type_ids=batch_seq_segments,
            labels=labels,
        )[:2]
        probabilities = nn.functional.softmax(logits, dim=-1)
        return loss, logits, probabilities


class RobertModel(nn.Module):
    def __init__(self, requires_grad=True):
        super(RobertModel, self).__init__()
        self.bert = RobertaForSequenceClassification.from_pretrained(
            "roberta-base", num_labels=2
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "roberta-base", do_lower_case=True
        )
        self.requires_grad = requires_grad
        self.device = torch.device("cuda")
        for param in self.bert.parameters():
            param.requires_grad = (
                requires_grad  # Each parameter requires gradient
            )

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments, labels):
        loss, logits = self.bert(
            input_ids=batch_seqs,
            attention_mask=batch_seq_masks,
            token_type_ids=batch_seq_segments,
            labels=labels,
        )[:2]
        probabilities = nn.functional.softmax(logits, dim=-1)
        return loss, logits, probabilities


class XlnetModel(nn.Module):
    def __init__(self, requires_grad=True):
        super(XlnetModel, self).__init__()
        self.xlnet = XLNetForSequenceClassification.from_pretrained(
            "xlnet-large-cased", num_labels=2
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "xlnet-large-cased", do_lower_case=True
        )
        self.requires_grad = requires_grad
        self.device = torch.device("cuda")
        for param in self.xlnet.parameters():
            param.requires_grad = (
                requires_grad  # Each parameter requires gradient
            )

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments, labels):
        loss, logits = self.xlnet(
            input_ids=batch_seqs,
            attention_mask=batch_seq_masks,
            token_type_ids=batch_seq_segments,
            labels=labels,
        )[:2]
        probabilities = nn.functional.softmax(logits, dim=-1)
        return loss, logits, probabilities

def model_train(
    texts,
    labels,
    target_dir="/Users/lechuanwang/Downloads/ForteAug/bertmodels",
    max_seq_len=50,
    epochs=3,
    batch_size=32,
    lr=2e-05,
    patience=1,
    max_grad_norm=10.0,
    if_save_model=True,
):
    model = BertModel(requires_grad=True)
    tokenizer = model.tokenizer
    train_data = ClassificationDataProcessor(
        tokenizer, texts, labels, max_seq_len=max_seq_len
    )
    train_loader = DataLoader(train_data, batch_size=batch_size)

    device = torch.device("cpu")
    model = model.to(device)
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in param_optimizer
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.85, patience=0
    )

    best_score = 0.0
    start_epoch = 1
    # Data for loss curves plot
    epochs_count = []
    train_losses = []
    train_accuracies = []
    valid_losses = []
    valid_accuracies = []
    valid_aucs = []

    # -------------------- Training epochs -----------------------------------#

    print(
        "\n",
        20 * "=",
        "Training bert model on device: {}".format(device),
        20 * "=",
    )
    patience_counter = 0
    for epoch in range(start_epoch, epochs + 1):
        epochs_count.append(epoch)

        print("* Training epoch {}:".format(epoch))
        epoch_time, epoch_loss, epoch_accuracy = train(
            model, train_loader, optimizer, epoch, max_grad_norm
        )
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        print(
            "-> Training time: {:.4f}s, loss = {:.4f}, accuracy: {:.4f}%".format(
                epoch_time, epoch_loss, (epoch_accuracy * 100)
            )
        )

        # Update the optimizer's learning rate with the scheduler.
        scheduler.step(epoch_accuracy)

        # Early stopping on validation accuracy.
        if epoch_accuracy < best_score:
            patience_counter += 1
        else:
            best_score = epoch_accuracy
            patience_counter = 0
            if if_save_model:
                torch.save(
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "best_score": best_score,
                        "epochs_count": epochs_count,
                        "train_losses": train_losses,
                        "train_accuracy": train_accuracies,
                        "valid_losses": valid_losses,
                        "valid_accuracy": valid_accuracies,
                        "valid_auc": valid_aucs,
                    },
                    os.path.join(target_dir, "best.pth.tar"),
                )
                print("save model succesfully!\n")

        if patience_counter >= patience:
            print("-> Early stopping: patience limit reached, stopping...")
            break
