if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
import dill
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
from termcolor import cprint
import time
import threading
from hydra.core.hydra_config import HydraConfig
from canonical_policy.policy.base_image_policy import BaseImagePolicy
from canonical_policy.dataset.base_dataset import BaseImageDataset
from canonical_policy.env_runner.base_image_runner import BaseImageRunner
from canonical_policy.common.checkpoint_util import TopKCheckpointManager
from canonical_policy.common.pytorch_util import dict_apply, optimizer_to
from canonical_policy.model.diffusion.ema_model import EMAModel
from canonical_policy.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainTemporalCanonicalWorkspace:
    include_keys = ['global_step', 'epoch']
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir=None):
        self.cfg = cfg
        self._output_dir = output_dir
        self._saving_thread = None
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.model: BaseImagePolicy = hydra.utils.instantiate(cfg.policy)
        self.ema_model: BaseImagePolicy = None
        if cfg.training.use_ema:
            try:
                self.ema_model = copy.deepcopy(self.model)
            except:
                self.ema_model = hydra.utils.instantiate(cfg.policy)
        self.optimizer = hydra.utils.instantiate(cfg.optimizer, params=self.model.parameters())
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        RUN_ROLLOUT = True
        RUN_CKPT = True
        RUN_VALIDATION = False
        dataset: BaseImageDataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)
        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(len(train_dataloader) * cfg.training.num_epochs) // cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step-1
        )
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)
        env_runner: BaseImageRunner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=self.output_dir)
        if env_runner is not None:
            assert isinstance(env_runner, BaseImageRunner)
        cfg.logging.name = str(cfg.logging.name)
        wandb_run = wandb.init(dir=str(self.output_dir), config=OmegaConf.to_container(cfg, resolve=True), **cfg.logging)
        wandb.config.update({"output_dir": self.output_dir})
        topk_manager = TopKCheckpointManager(save_dir=os.path.join(self.output_dir, 'checkpoints'), **cfg.checkpoint.topk)
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)
        # Use full precision (FP32) training; disable AMP entirely
        train_sampling_batch = None
        while self.epoch < cfg.training.num_epochs:
            step_log = dict()
            train_losses = list()
            with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                for batch_idx, batch in enumerate(tepoch):
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                    if train_sampling_batch is None:
                        train_sampling_batch = batch
                    raw_loss, loss_dict = self.model.compute_loss(batch)
                    loss = raw_loss / cfg.training.gradient_accumulate_every
                    loss.backward()
                    if self.global_step % cfg.training.gradient_accumulate_every == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)
                        lr_scheduler.step()
                    if cfg.training.use_ema:
                        ema.step(self.model)
                    raw_loss_cpu = raw_loss.item()
                    tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                    train_losses.append(raw_loss_cpu)
                    step_log = {'train_loss': raw_loss_cpu, 'global_step': self.global_step, 'epoch': self.epoch, 'lr': lr_scheduler.get_last_lr()[0]}
                    step_log.update(loss_dict)
                    is_last_batch = (batch_idx == (len(train_dataloader)-1))
                    if not is_last_batch:
                        wandb_run.log(step_log, step=self.global_step)
                        self.global_step += 1
                    if (cfg.training.max_train_steps is not None) and batch_idx >= (cfg.training.max_train_steps-1):
                        break
            train_loss = np.mean(train_losses)
            step_log['train_loss'] = train_loss
            policy = self.model
            if cfg.training.use_ema:
                policy = self.ema_model
            policy.eval()
            if ((self.epoch % cfg.training.rollout_every) == 0 and RUN_ROLLOUT and env_runner is not None) or (self.epoch > (cfg.training.num_epochs-10)):
                runner_log = env_runner.run(policy)
                step_log.update(runner_log)
            if (self.epoch % cfg.training.sample_every) == 0:
                with torch.no_grad():
                    batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                    obs_dict = batch['obs']
                    gt_action = batch['action']
                    result = policy.predict_action(obs_dict)
                    pred_action = result['action_pred']
                    mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                    step_log['train_action_mse_error'] = mse.item()
                    del batch, obs_dict, gt_action, result, pred_action, mse
            if env_runner is None:
                step_log['test_mean_score'] = - train_loss
            if (self.epoch % cfg.training.checkpoint_every) == 0 and cfg.checkpoint.save_ckpt:
                if cfg.checkpoint.save_last_ckpt:
                    self.save_checkpoint()
                if cfg.checkpoint.save_last_snapshot:
                    self.save_snapshot()
                metric_dict = {k.replace('/', '_'): v for k, v in step_log.items()}
                topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                if topk_ckpt_path is not None:
                    self.save_checkpoint(path=topk_ckpt_path)
            policy.train()
            wandb_run.log(step_log, step=self.global_step)
            self.global_step += 1
            self.epoch += 1
            del step_log

    @property
    def output_dir(self):
        output_dir = self._output_dir
        if output_dir is None:
            output_dir = HydraConfig.get().runtime.output_dir
        return output_dir

    def save_checkpoint(self, path=None, tag='latest', exclude_keys=None, include_keys=None, use_thread=False):
        if path is None:
            path = pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        else:
            path = pathlib.Path(path)
        if exclude_keys is None:
            exclude_keys = tuple(self.exclude_keys)
        if include_keys is None:
            include_keys = tuple(self.include_keys) + ('_output_dir',)
        path.parent.mkdir(parents=False, exist_ok=True)
        payload = {'cfg': self.cfg, 'state_dicts': dict(), 'pickles': dict()}
        for key, value in self.__dict__.items():
            if hasattr(value, 'state_dict') and hasattr(value, 'load_state_dict'):
                if key not in exclude_keys:
                    if use_thread:
                        payload['state_dicts'][key] = _copy_to_cpu(value.state_dict())
                    else:
                        payload['state_dicts'][key] = value.state_dict()
            elif key in include_keys:
                payload['pickles'][key] = dill.dumps(value)
        if use_thread:
            self._saving_thread = threading.Thread(target=lambda : torch.save(payload, path.open('wb'), pickle_module=dill))
            self._saving_thread.start()
        else:
            torch.save(payload, path.open('wb'), pickle_module=dill)
        del payload
        torch.cuda.empty_cache()
        return str(path.absolute())

    def get_checkpoint_path(self, tag='latest'):
        if tag=='latest':
            return pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        else:
            raise NotImplementedError

    def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
        if exclude_keys is None:
            exclude_keys = tuple()
        if include_keys is None:
            include_keys = payload['pickles'].keys()
        for key, value in payload['state_dicts'].items():
            if key not in exclude_keys:
                self.__dict__[key].load_state_dict(value, **kwargs)
        for key in include_keys:
            if key in payload['pickles']:
                self.__dict__[key] = dill.loads(payload['pickles'][key])

    def load_checkpoint(self, path=None, tag='latest', exclude_keys=None, include_keys=None, **kwargs):
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        else:
            path = pathlib.Path(path)
        payload = torch.load(path.open('rb'), pickle_module=dill, map_location='cpu')
        self.load_payload(payload, exclude_keys=exclude_keys, include_keys=include_keys)
        return payload

    def save_snapshot(self, tag='latest'):
        path = pathlib.Path(self.output_dir).joinpath('snapshots', f'{tag}.pkl')
        path.parent.mkdir(parents=False, exist_ok=True)
        torch.save(self, path.open('wb'), pickle_module=dill)
        return str(path.absolute())

    @classmethod
    def create_from_snapshot(cls, path):
        return torch.load(open(path, 'rb'), pickle_module=dill)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'canonical_policy', 'config'))
)
def main(cfg):
    workspace = TrainTemporalCanonicalWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main() 