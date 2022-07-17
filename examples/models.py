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
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

import numpy as np
from typing import List, Optional, Tuple, Union


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


class BertForSentimentAnalysis(BertForSequenceClassification):
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        input_ids_b: Optional[torch.Tensor] = None,
        attention_mask_b: Optional[torch.Tensor] = None,
        token_type_ids_b: Optional[torch.Tensor] = None,
        position_ids_b: Optional[torch.Tensor] = None,
        head_mask_b: Optional[torch.Tensor] = None,
        inputs_embeds_b: Optional[torch.Tensor] = None,
        labels_b: Optional[torch.Tensor] = None,
        output_attentions_b: Optional[bool] = None,
        output_hidden_states_b: Optional[bool] = None,
        return_dict_b: Optional[bool] = None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict
            if return_dict is not None
            else self.config.use_return_dict
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        outputs_b = self.bert(
            input_ids_b,
            attention_mask=attention_mask_b,
            token_type_ids=token_type_ids_b,
            position_ids=position_ids_b,
            head_mask=head_mask_b,
            inputs_embeds=inputs_embeds_b,
            output_attentions=output_attentions_b,
            output_hidden_states=output_hidden_states_b,
            return_dict=return_dict_b,
        )

        mix_rate = np.random.beta(8, 8)
        pooled_output = np.array(outputs[1]) * mix_rate + np.array(
            outputs_b[1]
        ) * (1 - mix_rate)
        labels = np.array(labels) * mix_rate + np.array(outputs_b[1]) * (
            1 - mix_rate
        )

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1)
                )
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return (
            loss,
            logits,
            outputs.hidden_states,
            outputs.attentions,
        )


class BertModel(nn.Module):
    def __init__(self, requires_grad=True):
        super(BertModel, self).__init__()
        self.bert = BertForSentimentAnalysis.from_pretrained(
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

    def forward(
        self,
        batch_seqs,
        batch_seq_masks,
        batch_seq_segments,
        labels,
        batch_seqs_b=None,
        batch_seq_masks_b=None,
        batch_seq_segments_b=None,
        labels_b=None,
    ):
        loss, logits = self.bert(
            input_ids=batch_seqs,
            attention_mask=batch_seq_masks,
            token_type_ids=batch_seq_segments,
            labels=labels,
            input_ids_b=batch_seqs_b,
            attention_mask_b=batch_seq_masks_b,
            token_type_ids_b=batch_seq_segments_b,
            labels_b=labels_b,
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
