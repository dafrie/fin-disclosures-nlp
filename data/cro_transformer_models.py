import os

import numpy as np

from sklearn.metrics import average_precision_score, accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, roc_auc_score, matthews_corrcoef
from sklearn.utils.class_weight import compute_class_weight
from transformers import logging, AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer
from transformers.trainer_pt_utils import nested_detach

import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class CroTrainer(Trainer):
    """Custom implementation of the "Trainer" class that adds default behaviour such as custom metrics,
    weights in the loss function and tokenization and initialization

    """

    def __init__(self, model_checkpoint=None, dataset=None, task=None, avg_strategy=None, max_token_size=256, should_weight=True, **kwargs):
        """Custom initialisation that calls the super class method of the base Trainer class.
        kwargs contains the TrainerArguments, so any additional parameter (that is not used in the base class) must be a named argument.
        Dataset doesn't need to be tokenized.
        """

        # Initialize custom stuff
        self.model_checkpoint = model_checkpoint
        if task == 'binary':
            self.num_labels = 2
        elif task == 'multi-class':
            self.num_labels = dataset['train'].features['labels'].num_classes
        else:
            self.num_labels = dataset['train'].features['labels'].feature.num_classes
        self.dataset = dataset
        self.task = task
        self.avg_strategy = avg_strategy
        self.weights = None

        if should_weight:
            self.weights = self.get_pos_weights()
            print(f"Using weights: {self.weights}")

        # Set and load the tokenizer.
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)

        # Tokenize the dataset
        print("Tokenization....")
        self.dataset = self.dataset.map(lambda ds: self.tokenizer(
            ds["text"], truncation=True, padding='max_length', max_length=max_token_size), batched=True)

        # Setup the trainer arguments
        self.training_args = TrainingArguments(**kwargs)

        # Initialize actual trainer
        super().__init__(
            model_init=self.my_model_init,  # Notice the custom implementation
            args=self.training_args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["valid"],
            tokenizer=self.tokenizer,
            compute_metrics=self.my_compute_metrics  # Notice the custom implementation
        )

    def my_model_init(self):
        """Custom model initialization. Disabels logging temporarily to avoid spamming messages and loads the pretrained or fine-tuned model"""
        logging.set_verbosity_error()  # Workaround to hide warnings that the model weights are randomly set and fine-tuning is necessary (which we do later...)
        model = AutoModelForSequenceClassification.from_pretrained(
            # Load from model checkpoint, i.e. the pretrained model or a previously saved fine-tuned model
            self.model_checkpoint,
            num_labels=self.num_labels,  # The number of different categories/labels
            # Whether the model returns attentions weights.
            output_attentions=False,
            output_hidden_states=False,  # Whether the model returns all hidden-states.)
            return_dict=False,  # TODO: Change this back and ammend the custom functions
        )
        logging.set_verbosity_warning()
        return model

    def prediction_step_(self, model, inputs, prediction_loss_only, ignore_keys):
        """Custom method overwriting the provided method to include "compute_loss()" which lead to an error when evaluating
          See previous PR: https://github.com/huggingface/transformers/pull/7074/files
        """

        # Return the inherited method if not absolutely necessary
        if self.task != 'multi-label':
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

        has_labels = all(inputs.get(k) is not None for k in self.label_names)

        # These two lines are added
        if has_labels:
            labels = tuple(inputs.get(name) for name in self.label_names)

        inputs = self._prepare_inputs(inputs)

        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(
                    self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with torch.no_grad():
            if self.args.fp16 and _use_native_amp:
                with autocast():
                    # These two lines are added
                    if has_labels:
                        loss = self.compute_loss(model, inputs)

                    outputs = model(**inputs)
            else:
                # These two lines are added
                if has_labels:
                    loss = self.compute_loss(model, inputs)

                outputs = model(**inputs)
            if has_labels:
                if isinstance(outputs, dict):
                    # loss = outputs["loss"].mean().detach()
                    loss = loss.mean().detach()
                    logits = tuple(v for k, v in outputs.items()
                                   if k not in ignore_keys + ["loss"])
                else:
                    # loss = outputs[0].mean().detach()
                    loss = loss.mean().detach()
                    # logits = outputs[1:]
                    logits = outputs[0:]
            else:
                loss = None
                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items()
                                   if k not in ignore_keys)
                else:
                    logits = outputs

                # TODO: Remove
                # Slicing so we get a tuple even if `outputs` is a `ModelOutput`.
                # logits = outputs[:]

            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index if has_labels else self.args.past_index - 1]
                # TODO: Remove
                # Remove the past from the logits.
                # logits = logits[: self.args.past_index - 1] + \
                #    logits[self.args.past_index:]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        if has_labels:
            labels = nested_detach(labels)
            # ORIG: labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None
        return (loss, logits, labels)

    def get_pos_weights(self):
        """Calculates weights for each class that relates to the ratio of positive to negative sample in each class"""
        if self.task == 'multi-label':
            pos_counts = np.sum(self.dataset['train']['labels'], axis=0)
            neg_counts = [self.dataset['train'].num_rows -
                          pos_count for pos_count in pos_counts]
            pos_weights = neg_counts / pos_counts

        else:
            y = self.dataset['train']['labels']
            pos_weights = compute_class_weight(
                class_weight='balanced', classes=np.unique(y), y=y)

        tensor = torch.as_tensor(pos_weights, dtype=torch.float)
        if torch.cuda.is_available():
            return tensor.to(device="cuda")
        return tensor

    def compute_loss(self, model, inputs, return_outputs=False):
        """Implements a BinaryCrossEntropyWithLogits activation and loss function
          to support multi-label cases
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        # logits = outputs[0]
        logits = outputs.logits

        ###############################
        #   Note: The multi-label conditional is added since we want to use a different loss function for this case
        ###############################
        # Multi-Label: Example: [0 1 1]
        if self.task == 'multi-label':
            # To adjust for inbalanced data, the pos_weight
            loss_func = BCEWithLogitsLoss(pos_weight=self.weights)
            #labels = labels.float()
            loss = loss_func(logits.view(-1, self.model.config.num_labels),  # model.num_labels),  # The logits
                             labels.float().view(-1, self.model.config.num_labels)  # model.num_labels)
                             )  # The labels
        # Binary or multi-class
        else:
            if model.num_labels == 1:
                loss_fct = MSELoss()  # Doing regression
                loss = loss_fct(logits.view(-1), labels.view(-1))
            # Multi-class (Example: [0 0 1]):
            else:
                loss_fct = CrossEntropyLoss(weight=self.weights)
                loss = loss_fct(
                    logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

    def my_compute_metrics(self, pred):
        """Computes classification task metric. Supports both multi-class and multi-label"""
        labels = pred.label_ids
        preds = pred.predictions

        # Multi-label
        if self.task == 'multi-label':
            roc_auc = roc_auc_score(labels, preds, average=self.avg_strategy)
            pr_auc = average_precision_score(
                labels, preds, average=self.avg_strategy)

            # TODO: Also add some threshold aware eval metrics, such as accuracy, ...
            # Problem: How do we set the threshold?

            return {
                'roc_auc': roc_auc,
                'pr_auc': pr_auc
            }

        # Binary/Multi-class task
        else:
            preds_bool = preds.argmax(-1)

            # Threshold unaware metrics
            # roc_auc = roc_auc_score(labels, preds, average=self.avg_strategy)
            # pr_auc = average_precision_score(labels, preds, average = self.avg_strategy)
            preds = preds.argmax(-1)

            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, preds_bool, average=self.avg_strategy)
            acc = accuracy_score(labels, preds_bool)
            balanced_accuracy = balanced_accuracy_score(
                labels, preds_bool, adjusted=True)
            matthews_corr = matthews_corrcoef(labels, preds_bool)
            return {
                # 'roc_auc': roc_auc,
                # 'pr_auc': pr_auc,
                'accuracy': acc,
                'balanced_accuracy': balanced_accuracy,
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'matthews_correlation': matthews_corr
            }
