import equinox as eqx
from typing import Dict, Any, Optional, Iterator, List
import jax
from .trainstate import TrainState
from collections import defaultdict
import time
from tqdm import tqdm
import optax
from jaxtyping import Array

# WANDBLOGGER = pl


class TrainerModule:
    def __init__(
        self,
        model: eqx.Module,
        optimizer: optax.GradientTransformation,
        input_init: Array,
        seed: int = 42,
        pl_logger: Optional=None,
        enable_progress_bar: bool = True,
        debug: bool = False,
        check_val_every_n_epoch: int = 1,
        log_dir: str = "",
        **kwargs,
    ):
        """
        A basic Trainer module summarizing most common training functionalities
        like logging, model initialization, training loop, etc.

        Atributes:
          model (eqx.Module): The class of the model that should be trained.
          optimizer: A dictionary of all hyperparameters of the optimizer.
            Used during initialization of the optimizer.
          exmp_input: Input to the model for initialization and tabulate.
          seed: Seed to initialize PRNG.
          logger_params: A dictionary containing the specification of the logger.
          enable_progress_bar: If False, no progress bar is shown.
          debug: If True, no jitting is applied. Can be helpful for debugging.
          check_val_every_n_epoch: The frequency with which the model is evaluated
            on the validation set.
        """
        super().__init__()
        self.optimizer = optimizer
        self.enable_progress_bar = enable_progress_bar
        self.debug = debug
        self.seed = seed
        self.check_val_every_n_epochs = check_val_every_n_epoch
        self.input_init = input_init
        self.log_dir = log_dir
        # init trainer parts
        self.pl_logger = pl_logger
        self.optimizer = optimizer
        self.create_jitted_functions()
        self.model = model
        self.optimizer = optimizer

    # def init_model(self, input_init: Any):
    #     model_rng = jax.random.PRNGKey(self.seed)
    #     model_rng, init_rng = jax.random.split(model_rng)
    #     self.run_model_init(input_init)

    def run_model_init(self, input_init):
        return jax.vmap(self.model)(input_init)

    def create_jitted_functions(self):
        train_step, eval_step = self.create_functions()
        if self.debug:
            self.train_step = train_step
            self.eval_step = eval_step

        else:
            self.train_step = eqx.filter_jit(train_step)
            # self.train_step = jax.jit(train_step)
            self.eval_step = eqx.filter_jit(eval_step)
            # self.eval_step = jax.jit(eval_step)

    def create_functions(self):
        def train_step(state: TrainState, batch: Any):
            metrics = {}
            return state, metrics

        def eval_step(state: TrainState, batch: Any):
            metrics = {}
            return metrics

        raise NotImplementedError

    def tracker(self, iterator: Iterator, **kwargs):
        if self.enable_progress_bar:
            return tqdm(iterator, **kwargs)
        else:
            return iterator

    def train_epoch(
        self, train_dataloader: Iterator, model, opt_state
    ) -> Dict[str, Any]:
        """
        Trains a model for one epoch.

        Args:
          train_loader: Data loader of the training set.

        Returns:
          A dictionary of the average training metrics over all batches
          for logging.
        """
        # Train model for one epoch, and log avg loss and accuracy
        metrics = defaultdict(float)
        num_train_steps = len(train_dataloader)
        start_time = time.time()
        for batch in self.tracker(train_dataloader, desc="Training", leave=False):
            # self.state, loss, step_metrics = self.train_step(self.state, batch)
            model, opt_state, loss, step_metrics = self.train_step(model, opt_state, batch)
            for key in step_metrics:
                metrics["train/" + key] += step_metrics[key] / num_train_steps
        metrics = {key: metrics[key].item() for key in metrics}
        metrics["epoch_time"] = time.time() - start_time
        return model, opt_state, metrics

    def eval_model(self, dataloader: Iterator, model, log_prefix=""):
        metrics = defaultdict(float)
        num_elements = 0
        for batch in dataloader:
            step_metrics = self.eval_step(model, batch)
            batch_size = (
                batch[0].shape[0]
                if isinstance(batch, (list, tuple))
                else batch.shape[0]
            )
            for key in step_metrics:
                metrics[key] += step_metrics[key] * batch_size

            num_elements += batch_size
        metrics = {
            (log_prefix + key): (metrics[key] / num_elements).item() for key in metrics
        }
        return metrics

    def train_model(self, dm, num_epochs: int = 500):
        dm.setup()
        self.on_training_start()
        model = self.model
        opt_state = self.optimizer.init(model)

        with tqdm(range(1, num_epochs + 1), desc="Epochs") as pbar_epoch:
            for epoch_idx in pbar_epoch:
                self.on_training_epoch_start(epoch_idx)
                
                model, opt_state, train_metrics = self.train_epoch(
                    dm.train_dataloader(), model, opt_state
                )
                
                pbar_epoch.set_description(
                    f"Epochs: {epoch_idx} | Loss: {train_metrics['train/loss']:.3e}"
                )
                if self.pl_logger:
                    self.pl_logger.log_metrics(train_metrics, step=epoch_idx)
                self.on_training_epoch_end(epoch_idx)

                if epoch_idx % self.check_val_every_n_epochs == 0:
                    eval_metrics = self.eval_model(
                        dm.val_dataloader(), model, log_prefix="val/"
                    )
                    self.on_validation_epoch_end(
                        epoch_idx, eval_metrics, dm.val_dataloader()
                    )
                    if self.pl_logger:
                        self.pl_logger.log_metrics(eval_metrics, step=epoch_idx)

        # with tqdm(range(1, num_epochs+1), desc="Epochs") as pbar_epoch:
        #     for epoch_idx in pbar_epoch:
        #         self.on_training_epoch_start(epoch_idx)
        #         train_metrics = self.train_epoch(dm.train_dataloader())
        #         pbar_epoch.set_description(f"Epochs: {epoch_idx} | Loss: {train_metrics['train/loss']:.3e}")
        #         self.on_training_epoch_end(epoch_idx)
        #
        #         if epoch_idx % self.check_val_every_n_epochs == 0:
        #             eval_metrics = self.eval_model(dm.val_dataloader(), log_prefix="val/")
        self.model = model
        self.opt_state = opt_state
        self.on_training_end()
        if self.pl_logger:
            self.pl_logger.finalize("success")
        return train_metrics

    def on_training_start(self):
        pass

    def on_training_epoch_start(self, epoch_idx: int):
        pass

    def on_training_epoch_end(self, epoch_idx: int):
        pass

    def on_validation_epoch_end(self, epoch_idx, eval_metrics, dataloader):
        pass

    def on_training_end(self):
        pass

    def save_model(self, name: str = "checkpoint.ckpt"):
        from pathlib import Path

        path = Path(self.log_dir).joinpath(name)
        eqx.tree_serialise_leaves(str(path), self.model)

    def load_model(self, name: str):
        model = eqx.tree_deserialise_leaves(f"{name}", self.model)
        self.model = model
