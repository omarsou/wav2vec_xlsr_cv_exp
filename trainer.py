import time
import torch
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import trainer_pt_utils
from transformers import set_seed
from transformers.trainer_utils import EvalPrediction, denumpify_detensorize, PredictionOutput
from transformers.trainer_pt_utils import DistributedTensorGatherer
from utils import generate_per_score, compute_wer


class Trainer:
    def __init__(self, model, processor, optimizer, warmup_scheduler, validation_freq, log_freq, num_warmup_steps,
                 is_plateau_scheduler=False, scheduler_on_plateau_freq=None, plateau_lr_params=None, type_score='PER'):
        self.model = model
        self.processor = processor
        self.optimizer = optimizer
        self.warmup_scheduler = warmup_scheduler
        self.validation_freq = validation_freq
        self.log_freq = log_freq
        self.num_warmup_steps = num_warmup_steps
        self.is_plateau_scheduler = is_plateau_scheduler
        if is_plateau_scheduler:
            self.scheduler_on_plateau_freq = scheduler_on_plateau_freq
            self.factor, self.patience = plateau_lr_params
        self.plateau_scheduler = None
        self.type_score = type_score

    def train(self, train_loader, valid_loader, n_epochs=350, path=None, seed=42):
        self.model.zero_grad()
        set_seed(seed)
        current_total_steps = 0
        current_best_metric = float('inf')
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Metric':^9} | "
              f"{'Elapsed':^9}")
        print("-" * 70)

        for epoch in range(n_epochs):
            t0_epoch, t0_freq = time.time(), time.time()
            tr_loss = 0.0
            epoch_step = 0

            for step, inputs in enumerate(train_loader):
                self.model.train()
                inputs = self.prepare_inputs(inputs)
                loss = self.compute_loss(inputs)

                loss.backward()

                tr_loss += loss.item()

                if hasattr(self.optimizer, "clip_grad_norm"):
                    self.optimizer.clip_grad_norm(1.0)
                elif hasattr(self.model, "clip_grad_norm"):
                    self.model.clip_grad_norm(1.0)

                self.optimizer.step()

                current_total_steps += 1
                epoch_step += 1

                if current_total_steps < self.num_warmup_steps + 1 or not self.is_plateau_scheduler:
                    self.warmup_scheduler.step()

                if current_total_steps % self.log_freq == 0:
                    time_elapsed = time.time() - t0_freq
                    print(
                        f"{epoch + 1:^7} | {epoch_step:^7} | {tr_loss / epoch_step:^12.6f} | {'-':^9} | "
                        f"{time_elapsed:^9.2f}")

                # Initialize the lr_on_plateau as soon as we have finished the warmup
                if self.is_plateau_scheduler and self.plateau_scheduler is None and current_total_steps > \
                        self.num_warmup_steps + 1:
                    self.plateau_scheduler = ReduceLROnPlateau(self.optimizer, factor=self.factor,
                                                               patience=self.patience, verbose=1)

                self.model.zero_grad()

                if current_total_steps % self.validation_freq == 0:
                    world_size = 1
                    score = self.prediction_loop(valid_loader, world_size)
                    time_elapsed = time.time() - t0_freq
                    print(
                        f"{epoch + 1:^7} | {epoch_step:^7} | {tr_loss / epoch_step:^12.6f} | {score:^9.2f} | "
                        f"{time_elapsed:^9.2f}")

                    if score < current_best_metric:
                        print("Hooray! New Best Validation Score, Saving model.")
                        torch.save(self.model.state_dict(), path)
                        current_best_metric = score

    def compute_loss(self, inputs, return_outputs=False):
        outputs = self.model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        return (loss, outputs) if return_outputs else loss

    def prediction_loop(self, data_loader, world_size):
        num_examples = len(data_loader.dataset)
        batch_size = data_loader.batch_size
        eval_losses_gatherer = DistributedTensorGatherer(world_size, num_examples,
                                                         make_multiple_of=batch_size)
        preds_gatherer = DistributedTensorGatherer(world_size, num_examples)
        labels_gatherer = DistributedTensorGatherer(world_size, num_examples)
        losses_host, preds_host, labels_host = None, None, None
        self.model.eval()

        for step, inputs in enumerate(data_loader):
            loss, logits, labels = self.prediction_step(inputs)
            losses = loss.repeat(batch_size)
            losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            preds_host = logits if preds_host is None else trainer_pt_utils.nested_concat(preds_host, logits,
                                                                                          padding_index=-100)
            labels_host = labels if labels_host is None else trainer_pt_utils.nested_concat(labels_host, labels,
                                                                                            padding_index=-100)
            eval_losses_gatherer.add_arrays(trainer_pt_utils.nested_numpify(losses_host))
            preds_gatherer.add_arrays(trainer_pt_utils.nested_numpify(preds_host))
            labels_gatherer.add_arrays(trainer_pt_utils.nested_numpify(labels_host))
            losses_host, preds_host, labels_host = None, None, None

        eval_loss = eval_losses_gatherer.finalize()
        preds = preds_gatherer.finalize()
        labels_ids = labels_gatherer.finalize()

        if self.type_score == "PER":
            preds_ids = np.argmax(preds, axis=-1)

            predicted_phonemes = self.processor.batch_decode(torch.from_numpy(preds_ids))
            true_phonemes = self.processor.batch_decode(torch.from_numpy(labels_ids))

            per = generate_per_score(true_phonemes, predicted_phonemes)

            return per

        elif self.type_score == "WER":
            pred = EvalPrediction(predictions=preds, label_ids=labels_ids)
            pred_logits = pred.predictions
            pred_ids = np.argmax(pred_logits, axis=-1)

            pred.label_ids[pred.label_ids == -100] = self.processor.tokenizer.pad_token_id

            pred_str = self.processor.batch_decode(pred_ids)

            # we do not want to group tokens when computing the metrics
            label_str = self.processor.batch_decode(pred.label_ids, group_tokens=False)

            metrics = compute_wer(pred_str, label_str)
            metrics = denumpify_detensorize(metrics)
            metrics["t_loss"] = eval_loss.mean().item()
            wer = PredictionOutput(preds, labels_ids, metrics).metrics["wer"]

            return wer

    def prediction_step(self, inputs, label_names=["labels"]):
        has_labels = all(inputs.get(k) is not None for k in label_names)
        inputs = self.prepare_inputs(inputs)
        if hasattr(self.model, "config"):
            ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
        else:
            ignore_keys = []

        if has_labels:
            labels = trainer_pt_utils.nested_detach(tuple(inputs.get(name) for name in label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if has_labels:
                loss, outputs = self.compute_loss(inputs, True)
                loss = loss.mean().detach()
                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                else:
                    logits = outputs[1:]
            else:
                loss, outputs = None, self.model(**inputs)
                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                else:
                    logits = outputs

        logits = trainer_pt_utils.nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)

    def compute_score(self, data_loader, path_to_load):
        self.model.load_state_dict(torch.load(path_to_load))
        return self.prediction_loop(data_loader, 1)

    @staticmethod
    def prepare_inputs(inputs):
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.cuda()
        return inputs


