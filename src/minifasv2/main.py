import torch
from torch import optim
from torch.nn import CrossEntropyLoss, MSELoss
from tensorboardX import SummaryWriter
import os
import json
import time
import numpy as np
from sklearn.metrics import confusion_matrix

from src.minifasv2.model import MultiFTNet
from src.minifasv2.data import get_train_valid


class Trainer:
    def __init__(self, config, resume_from=None, transfer_learning=False):
        self.config = config
        self.step = 0
        self.validation_step = 0
        self.start_epoch = 0
        self.best_validation_accuracy = 0.0
        self.resume_from = resume_from
        self.transfer_learning = transfer_learning
        self.train_loader, self.valid_loader = get_train_valid(self.config)
        self.log_train_every = len(self.train_loader) // config.log_loss_per_epoch
        self.log_valid_every = len(self.valid_loader) // config.log_loss_per_epoch

    def train_model(self):
        self._init_model_param()
        if self.resume_from and os.path.exists(self.resume_from):
            if self.transfer_learning:
                self._load_checkpoint_for_transfer(self.resume_from)
            else:
                self._load_checkpoint(self.resume_from)
        self._train_stage()

    def _init_model_param(self):
        self.classification_criterion = CrossEntropyLoss()
        self.fourier_criterion = MSELoss()
        self.model = self._define_network()
        self.optimizer = optim.SGD(
            self.model.module.parameters(),
            lr=self.config.lr,
            weight_decay=5e-4,
            momentum=self.config.momentum,
        )

        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, self.config.milestones, self.config.gamma, -1
        )

        print("lr: ", self.config.lr)
        print("epochs: ", self.config.epochs)
        print("milestones: ", self.config.milestones)

    def _train_stage(self):
        if self.start_epoch == 0:
            self.writer = SummaryWriter(self.config.log_path)
        else:
            self.writer = SummaryWriter(self.config.log_path)

        total_epochs = self.config.epochs - self.start_epoch
        print(f'\n{"="*80}')
        print(
            f"Training: {self.start_epoch}/{self.config.epochs} epochs | LR: {self.config.lr} | Batch: {self.config.batch_size}"
        )
        print(f'{"="*80}\n')

        for e in range(self.start_epoch, self.config.epochs):
            epoch_start_time = time.time()
            epoch_progress = ((e - self.start_epoch + 1) / total_epochs) * 100

            print(f'\n{"="*80}')
            print(
                f"EPOCH [{e+1}/{self.config.epochs}] | Progress: {epoch_progress:.1f}% | LR: {self.lr_scheduler.get_last_lr()[0]:.6f}"
            )
            print(f'{"="*80}')

            self.model.train()
            train_loss = 0.0
            train_accuracy = 0.0
            train_classification_loss = 0.0
            train_fourier_loss = 0.0
            train_count = 0

            for batch_idx, (sample, fourier_sample, labels) in enumerate(
                self.train_loader
            ):
                imgs = [sample, fourier_sample]
                loss, accuracy, classification_loss, fourier_loss = (
                    self._train_batch_data(imgs, labels)
                )

                train_loss += loss
                train_accuracy += accuracy
                train_classification_loss += classification_loss
                train_fourier_loss += fourier_loss
                train_count += 1
                self.step += 1

                if self.step % self.log_train_every == 0 and self.step != 0:
                    log_step = self.step // self.log_train_every
                    self.writer.add_scalar(
                        "Loss/train", train_loss / train_count, log_step
                    )
                    self.writer.add_scalar(
                        "Acc/train", train_accuracy / train_count, log_step
                    )
                    self.writer.add_scalar(
                        "Loss_cls/train",
                        train_classification_loss / train_count,
                        log_step,
                    )
                    self.writer.add_scalar(
                        "Loss_ft/train", train_fourier_loss / train_count, log_step
                    )
                    self.writer.add_scalar(
                        "Learning_rate",
                        self.optimizer.param_groups[0]["lr"],
                        log_step,
                    )

                if (
                    self.step % max(100, len(self.train_loader) // 4) == 0
                    and batch_idx > 0
                ):
                    self._save_mid_epoch_checkpoint(
                        e,
                        batch_idx,
                        train_loss / train_count,
                        train_accuracy / train_count,
                        train_classification_loss / train_count,
                        train_fourier_loss / train_count,
                    )

                batch_progress = ((batch_idx + 1) / len(self.train_loader)) * 100
                if (
                    batch_idx % max(10, len(self.train_loader) // 20) == 0
                    or batch_idx == len(self.train_loader) - 1
                ):
                    print(
                        f"\rTrain [{e+1}/{self.config.epochs}] | Progress: {batch_progress:5.1f}% | "
                        f"Loss: {train_loss/train_count:.4f} | LossCls: {train_classification_loss/train_count:.4f} | "
                        f"LossFT: {train_fourier_loss/train_count:.4f} | Acc: {train_accuracy/train_count:.4f}",
                        end="",
                        flush=True,
                    )

            self.lr_scheduler.step()

            avg_train_loss = train_loss / train_count if train_count > 0 else 0.0
            avg_train_accuracy = (
                train_accuracy / train_count if train_count > 0 else 0.0
            )
            avg_train_classification_loss = (
                train_classification_loss / train_count if train_count > 0 else 0.0
            )
            avg_train_fourier_loss = (
                train_fourier_loss / train_count if train_count > 0 else 0.0
            )

            self.model.eval()
            validation_accuracy = 0.0
            validation_classification_loss = 0.0
            validation_count = 0
            all_validation_preds = []
            all_validation_labels = []

            for batch_idx, (sample, labels) in enumerate(self.valid_loader):
                with torch.no_grad():
                    accuracy, classification_loss, preds = self._valid_batch_data(
                        sample, labels
                    )
                validation_accuracy += accuracy
                validation_classification_loss += classification_loss
                validation_count += 1
                self.validation_step += 1

                all_validation_preds.extend(preds.cpu().numpy())
                all_validation_labels.extend(labels.cpu().numpy())

                if (
                    self.validation_step % self.log_valid_every == 0
                    and self.validation_step != 0
                ):
                    log_step = self.validation_step // self.log_valid_every
                    self.writer.add_scalar(
                        "Acc/valid", validation_accuracy / validation_count, log_step
                    )
                    self.writer.add_scalar(
                        "Loss_cls/valid",
                        validation_classification_loss / validation_count,
                        log_step,
                    )

                validation_progress = ((batch_idx + 1) / len(self.valid_loader)) * 100
                if (
                    batch_idx % max(5, len(self.valid_loader) // 20) == 0
                    or batch_idx == len(self.valid_loader) - 1
                ):
                    avg_validation_loss = (
                        validation_classification_loss / validation_count
                    )
                    avg_validation_accuracy = validation_accuracy / validation_count
                    print(
                        f"\rValid [{e+1}/{self.config.epochs}] | Progress: {validation_progress:5.1f}% | "
                        f"Loss: {avg_validation_loss:.4f} | Val Acc: {avg_validation_accuracy*100:.2f}%",
                        end="",
                        flush=True,
                    )

            avg_validation_accuracy = (
                validation_accuracy / validation_count if validation_count > 0 else 0.0
            )
            avg_validation_loss = (
                validation_classification_loss / validation_count
                if validation_count > 0
                else 0.0
            )

            all_validation_preds = np.array(all_validation_preds)
            all_validation_labels = np.array(all_validation_labels)
            num_classes = self.config.num_classes
            confusion_matrix_result = confusion_matrix(
                all_validation_labels,
                all_validation_preds,
                labels=list(range(num_classes)),
            )
            per_class_accuracy = []
            for i in range(num_classes):
                if confusion_matrix_result[i, :].sum() > 0:
                    per_class_accuracy.append(
                        confusion_matrix_result[i, i]
                        / confusion_matrix_result[i, :].sum()
                    )
                else:
                    per_class_accuracy.append(0.0)

            is_best = avg_validation_accuracy > self.best_validation_accuracy
            if is_best:
                self.best_validation_accuracy = avg_validation_accuracy

            epoch_time = time.time() - epoch_start_time
            total_progress = ((e - self.start_epoch + 1) / total_epochs) * 100
            remaining_epochs = self.config.epochs - (e + 1)
            eta_seconds = epoch_time * remaining_epochs
            eta_hours = int(eta_seconds // 3600)
            eta_mins = int((eta_seconds % 3600) // 60)

            print(f'\n\n{"="*80}')
            print(
                f"EPOCH [{e+1}/{self.config.epochs}] SUMMARY | Progress: {total_progress:.1f}% | ETA: {eta_hours}h {eta_mins}m"
            )
            print(f'{"="*80}')
            print(
                f"TRAIN: Loss={avg_train_loss:.4f} | LossCls={avg_train_classification_loss:.4f} | LossFT={avg_train_fourier_loss:.4f} | Train Acc={avg_train_accuracy*100:.2f}%"
            )
            print(
                f"VALID: Loss={avg_validation_loss:.4f} | LossCls={avg_validation_loss:.4f} | Val Acc={avg_validation_accuracy*100:.2f}% | Best Val Acc={self.best_validation_accuracy*100:.2f}%"
            )
            print(
                f"PER-CLASS VALID ACC: Real={per_class_accuracy[0]*100:.2f}% | Spoof={per_class_accuracy[1]*100:.2f}%"
            )
            print(
                f"TIME: {epoch_time:.1f}s | LR: {self.lr_scheduler.get_last_lr()[0]:.6f}"
            )
            if is_best:
                print(
                    f"★ NEW BEST MODEL! Val Acc improved to {avg_validation_accuracy*100:.2f}%"
                )
            print(f'{"="*80}\n')

            self._save_checkpoint(
                e,
                avg_validation_accuracy,
                avg_validation_loss,
                is_best,
                avg_train_accuracy,
                avg_train_loss,
                avg_train_classification_loss,
                avg_train_fourier_loss,
            )

        self.writer.close()

    def _train_batch_data(self, imgs, labels):
        self.optimizer.zero_grad()
        labels = labels.to(self.config.device)
        embeddings, feature_map = self.model.forward(imgs[0].to(self.config.device))

        classification_loss = self.classification_criterion(embeddings, labels)
        fourier_loss = self.fourier_criterion(
            feature_map, imgs[1].to(self.config.device)
        )

        loss = 0.5 * classification_loss + 0.5 * fourier_loss
        accuracy = self._get_accuracy(embeddings, labels)[0]
        loss.backward()
        self.optimizer.step()
        return (
            loss.item(),
            accuracy.item(),
            classification_loss.item(),
            fourier_loss.item(),
        )

    def _valid_batch_data(self, img, labels):
        labels = labels.to(self.config.device)
        embeddings = self.model.forward(img.to(self.config.device))

        classification_loss = self.classification_criterion(embeddings, labels)
        accuracy = self._get_accuracy(embeddings, labels)[0]
        preds = torch.argmax(embeddings, dim=1)

        return accuracy.item(), classification_loss.item(), preds

    def _define_network(self):
        param = {
            "num_classes": self.config.num_classes,
            "num_channels": self.config.num_channels,
            "embedding_size": self.config.embedding_size,
            "conv6_kernel": self.config.kernel_size,
        }

        model = MultiFTNet(**param).to(self.config.device)
        model = torch.nn.DataParallel(model)
        model.to(self.config.device)
        return model

    def _get_accuracy(self, output, target, topk=(1,)):
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        ret = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(dim=0, keepdim=True)
            ret.append(correct_k.mul_(1.0 / batch_size))
        return ret

    def _save_checkpoint(
        self,
        epoch,
        validation_accuracy,
        validation_loss,
        is_best=False,
        train_accuracy=0.0,
        train_loss=0.0,
        train_classification_loss=0.0,
        train_fourier_loss=0.0,
    ):
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.lr_scheduler.state_dict(),
            "step": self.step,
            "validation_step": self.validation_step,
            "best_validation_accuracy": self.best_validation_accuracy,
            "validation_accuracy": validation_accuracy,
            "validation_loss": validation_loss,
            "train_accuracy": train_accuracy,
            "train_loss": train_loss,
            "train_classification_loss": train_classification_loss,
            "train_fourier_loss": train_fourier_loss,
            "config": {
                "lr": self.config.lr,
                "batch_size": self.config.batch_size,
                "input_size": self.config.input_size,
                "num_classes": self.config.num_classes,
                "epochs": self.config.epochs,
                "milestones": self.config.milestones,
            },
        }

        checkpoint_path = os.path.join(self.config.model_path, "checkpoint_latest.pth")
        torch.save(checkpoint, checkpoint_path)

        epoch_checkpoint_path = os.path.join(
            self.config.model_path, f"checkpoint_epoch_{epoch:03d}.pth"
        )
        torch.save(checkpoint, epoch_checkpoint_path)

        if is_best:
            best_checkpoint_path = os.path.join(
                self.config.model_path, "checkpoint_best.pth"
            )
            torch.save(checkpoint, best_checkpoint_path)

        checkpoint_info = {
            "epoch": epoch,
            "train_accuracy": float(train_accuracy),
            "train_loss": float(train_loss),
            "train_classification_loss": float(train_classification_loss),
            "train_fourier_loss": float(train_fourier_loss),
            "validation_accuracy": float(validation_accuracy),
            "validation_loss": float(validation_loss),
            "best_validation_accuracy": float(self.best_validation_accuracy),
            "step": self.step,
            "validation_step": self.validation_step,
        }

        info_path = os.path.join(self.config.model_path, "training_info.json")
        with open(info_path, "w") as f:
            json.dump(checkpoint_info, f, indent=2)

    def _load_checkpoint_for_transfer(self, checkpoint_path):
        print(f"Loading checkpoint for TRANSFER LEARNING from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        new_state_dict = {}
        for key, value in state_dict.items():
            if "FTGenerator" not in key:
                name_key = key.replace("module.model.", "", 1).replace("module.", "", 1)
                name_key = name_key.replace("model.prob", "model.logits")
                name_key = name_key.replace(".prob", ".logits")
                name_key = name_key.replace("model.drop", "model.dropout")
                name_key = name_key.replace(".drop", ".dropout")
                new_state_dict[name_key] = value

        self.model.load_state_dict(new_state_dict, strict=False)

        self.start_epoch = 0
        self.step = 0
        self.validation_step = 0
        self.best_validation_accuracy = 0.0

        print(f"✅ Loaded model weights | Starting from EPOCH 1 | LR: {self.config.lr}")
        print("⚠️  Optimizer and scheduler reset for fresh training")

    def _load_checkpoint(self, checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)

        model_state_dict = checkpoint["model_state_dict"]
        new_model_state_dict = {}
        for key, value in model_state_dict.items():
            new_key = key
            if new_key.startswith("module."):
                new_key = new_key[7:]
            new_key = new_key.replace("model.prob", "model.logits")
            new_key = new_key.replace(".prob", ".logits")
            new_key = new_key.replace("model.drop", "model.dropout")
            new_key = new_key.replace(".drop", ".dropout")
            new_key = new_key.replace(
                "FTGenerator.ft.", "FTGenerator.fourier_transform."
            )
            new_key = new_key.replace("FTGenerator.ft", "FTGenerator.fourier_transform")
            new_model_state_dict[new_key] = value

        self.model.load_state_dict(new_model_state_dict, strict=False)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        saved_epoch = checkpoint["epoch"]
        if checkpoint.get("batch_idx", None) is not None:
            self.start_epoch = saved_epoch
        else:
            self.start_epoch = saved_epoch - 1

        self.step = checkpoint.get("step", 0)
        self.validation_step = checkpoint.get(
            "validation_step", checkpoint.get("val_step", 0)
        )
        self.best_validation_accuracy = checkpoint.get(
            "best_validation_accuracy", checkpoint.get("best_val_acc", 0.0)
        )

        batch_idx = checkpoint.get("batch_idx", None)
        if batch_idx is not None:
            print(
                f"Resumed from epoch {self.start_epoch + 1} (0-indexed: {self.start_epoch}), batch {batch_idx}, step {self.step}, best_validation_accuracy: {self.best_validation_accuracy:.4f}"
            )
            print(
                f"Note: Will restart epoch {self.start_epoch + 1} from beginning (mid-epoch resume not implemented)"
            )
        else:
            print(
                f"Resumed from epoch {self.start_epoch + 1} (0-indexed: {self.start_epoch}), step {self.step}, best_validation_accuracy: {self.best_validation_accuracy:.4f}"
            )

    def _save_mid_epoch_checkpoint(
        self,
        epoch,
        batch_idx,
        avg_loss,
        avg_accuracy,
        avg_classification_loss,
        avg_fourier_loss,
    ):
        checkpoint = {
            "epoch": epoch,
            "batch_idx": batch_idx,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.lr_scheduler.state_dict(),
            "step": self.step,
            "validation_step": self.validation_step,
            "best_validation_accuracy": self.best_validation_accuracy,
            "train_accuracy": avg_accuracy,
            "train_loss": avg_loss,
            "train_classification_loss": avg_classification_loss,
            "train_fourier_loss": avg_fourier_loss,
            "config": {
                "lr": self.config.lr,
                "batch_size": self.config.batch_size,
                "input_size": self.config.input_size,
                "num_classes": self.config.num_classes,
                "epochs": self.config.epochs,
                "milestones": self.config.milestones,
            },
        }
        checkpoint_path = os.path.join(self.config.model_path, "checkpoint_latest.pth")
        torch.save(checkpoint, checkpoint_path)
