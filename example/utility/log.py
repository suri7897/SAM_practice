import csv
import time
import os
from utility.loading_bar import LoadingBar

class Log:
    def __init__(self, log_each: int, initial_epoch=-1, type = 'singlestep', model='cnn'):
        self.loading_bar = LoadingBar(length=27)
        self.best_accuracy = 0.0
        self.log_each = log_each
        self.epoch = initial_epoch
        
        base_dir = os.path.dirname(os.path.dirname(__file__))
        results_dir = os.path.join(base_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        if model == 'cnn':
            results_dir = os.path.join(results_dir, "cnn")
        else :
            results_dir = os.path.join(results_dir, 'wrn')
        os.makedirs(results_dir, exist_ok=True)

        if type == 'singlestep':
            self.csv_path = os.path.join(results_dir, f"{model}_SAM_result.csv")
        elif type == 'multistep':
            self.csv_path = os.path.join(results_dir, f"{model}_MultiSAM_result.csv")
        else :
            self.csv_path = os.path.join(results_dir, f"{model}_SGD_result.csv")

        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "train_accuracy",
                             "valid_loss", "valid_accuracy", "learning_rate"])

        self.train_loss_epoch = 0.0
        self.train_acc_epoch = 0.0
        self.valid_loss_epoch = 0.0
        self.valid_acc_epoch = 0.0

    def train(self, len_dataset: int) -> None:
        self.epoch += 1
        if self.epoch == 0:
            self._print_header()
        else:
            self.flush()

        self.is_train = True
        self.last_steps_state = {"loss": 0.0, "accuracy": 0.0, "steps": 0}
        self._reset(len_dataset)

    def eval(self, len_dataset: int) -> None:
        self.flush()
        self.is_train = False
        self._reset(len_dataset)

    def __call__(self, model, loss, accuracy, learning_rate: float = None) -> None:
        if self.is_train:
            self._train_step(model, loss, accuracy, learning_rate)
        else:
            self._eval_step(loss, accuracy)

    def flush(self) -> None:
        loss = self.epoch_state["loss"] / self.epoch_state["steps"]
        acc = self.epoch_state["accuracy"] / self.epoch_state["steps"]

        if self.is_train:
            self.train_loss_epoch = loss
            self.train_acc_epoch = acc
            self.learning_rate_epoch = self.learning_rate

            print(
                f"\r┃{self.epoch:12d}  ┃{loss:12.4f}  │{100*acc:10.2f} %  ┃"
                f"{self.learning_rate:12.3e}  │{self._time():>12}  ┃",
                end="", flush=True,
            )
        else:
            self.valid_loss_epoch = loss
            self.valid_acc_epoch = acc
            print(f"{loss:12.4f}  │{100*acc:10.2f} %  ┃", flush=True)


            with open(self.csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.epoch,
                    f"{self.train_loss_epoch:.6f}",
                    f"{self.train_acc_epoch:.6f}",
                    f"{self.valid_loss_epoch:.6f}",
                    f"{self.valid_acc_epoch:.6f}",
                    f"{self.learning_rate_epoch:.6e}",
                ])

            if acc > self.best_accuracy:
                self.best_accuracy = acc

    def _train_step(self, model, loss, accuracy, learning_rate: float) -> None:
        self.learning_rate = learning_rate
        self.last_steps_state["loss"] += loss.sum().item()
        self.last_steps_state["accuracy"] += accuracy.sum().item()
        self.last_steps_state["steps"] += loss.size(0)
        self.epoch_state["loss"] += loss.sum().item()
        self.epoch_state["accuracy"] += accuracy.sum().item()
        self.epoch_state["steps"] += loss.size(0)
        self.step += 1

        if self.step % self.log_each == self.log_each - 1:
            loss = self.last_steps_state["loss"] / self.last_steps_state["steps"]
            accuracy = self.last_steps_state["accuracy"] / self.last_steps_state["steps"]

            self.last_steps_state = {"loss": 0.0, "accuracy": 0.0, "steps": 0}
            progress = self.step / self.len_dataset

            print(
                f"\r┃{self.epoch:12d}  ┃{loss:12.4f}  │{100*accuracy:10.2f} %  ┃{learning_rate:12.3e}  │{self._time():>12}  {self.loading_bar(progress)}",
                end="",
                flush=True,
            )

    def _eval_step(self, loss, accuracy) -> None:
        self.epoch_state["loss"] += loss.sum().item()
        self.epoch_state["accuracy"] += accuracy.sum().item()
        self.epoch_state["steps"] += loss.size(0)

    def _reset(self, len_dataset: int) -> None:
        self.start_time = time.time()
        self.step = 0
        self.len_dataset = len_dataset
        self.epoch_state = {"loss": 0.0, "accuracy": 0.0, "steps": 0}

    def _time(self) -> str:
        time_seconds = int(time.time() - self.start_time)
        return f"{time_seconds // 60:02d}:{time_seconds % 60:02d} min"

    def _print_header(self) -> None:
        print(f"┏━━━━━━━━━━━━━━┳━━━━━━━╸T╺╸R╺╸A╺╸I╺╸N╺━━━━━━━┳━━━━━━━╸S╺╸T╺╸A╺╸T╺╸S╺━━━━━━━┳━━━━━━━╸V╺╸A╺╸L╺╸I╺╸D╺━━━━━━━┓")
        print(f"┃              ┃              ╷              ┃              ╷              ┃              ╷              ┃")
        print(f"┃       epoch  ┃        loss  │    accuracy  ┃        l.r.  │     elapsed  ┃        loss  │    accuracy  ┃")
        print(f"┠──────────────╂──────────────┼──────────────╂──────────────┼──────────────╂──────────────┼──────────────┨")
