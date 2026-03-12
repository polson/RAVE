import hashlib
import os
import sys
from typing import Any, Dict, Optional

import gin
import lightning.pytorch as pl
import torch
import torch.nn.functional as F
import torchaudio
from absl import flags, app
from torch.utils.data import DataLoader

try:
    import rave
except:
    import sys, os 
    sys.path.append(os.path.abspath('.'))
    import rave

import rave
import rave.core
import rave.dataset
from rave.transforms import get_augmentations, add_augmentation


FLAGS = flags.FLAGS

flags.DEFINE_string('name', None, help='Name of the run', required=True)
flags.DEFINE_multi_string('config',
                          default='v2.gin',
                          help='RAVE configuration to use')
flags.DEFINE_multi_string('augment',
                           default = [],
                            help = 'augmentation configurations to use')
flags.DEFINE_string('db_path',
                    None,
                    help='Dataset path (LMDB root or MUSDB root)',
                    required=True)
flags.DEFINE_enum('dataset_format',
                  'lmdb',
                  ['lmdb', 'musdb'],
                  help='Dataset format: preprocessed LMDB or direct MUSDB stems')
flags.DEFINE_string('val_db_path',
                    None,
                    help='Optional explicit validation dataset root')
flags.DEFINE_string('musdb_stem',
                    'vocals.wav',
                    help='Stem filename used in MUSDB mode')
flags.DEFINE_string('out_path',
                     default="runs/",
                     help='Output folder')
flags.DEFINE_integer('max_steps',
                     6000000,
                     help='Maximum number of training steps')
flags.DEFINE_integer('val_every', 10000, help='Checkpoint model every n steps')
flags.DEFINE_integer('save_every',
                     500000,
                     help='save every n steps (default: just last)')
flags.DEFINE_integer('n_signal',
                     131072,
                     help='Number of audio samples to use during training')
flags.DEFINE_integer('channels', 0, help="number of audio channels")
flags.DEFINE_integer('batch', 8, help='Batch size')
flags.DEFINE_string('ckpt',
                    None,
                    help='Path to previous checkpoint of the run')
flags.DEFINE_multi_string('override', default=[], help='Override gin binding')
flags.DEFINE_integer('workers',
                     default=8,
                     help='Number of workers to spawn for dataset loading')
flags.DEFINE_multi_integer('gpu', default=None, help='GPU to use')
flags.DEFINE_bool('derivative',
                  default=False,
                  help='Train RAVE on the derivative of the signal')
flags.DEFINE_bool('normalize',
                  default=False,
                  help='Train RAVE on normalized signals')
flags.DEFINE_list('rand_pitch',
                  default=None,
                  help='activates random pitch')
flags.DEFINE_float('ema',
                   default=None,
                   help='Exponential weight averaging factor (optional)')
flags.DEFINE_bool('progress',
                  default=True,
                  help='Display training progress bar')
flags.DEFINE_bool('smoke_test', 
                  default=False,
                  help="Run training with n_batches=1 to test the model")
flags.DEFINE_bool('wandb',
                  default=False,
                  help='Enable Weights & Biases logging')
flags.DEFINE_string('wandb_project',
                    default='rave',
                    help='W&B project name')
flags.DEFINE_string('wandb_entity',
                    default=None,
                    help='Optional W&B entity/team')
flags.DEFINE_bool('wandb_save_code',
                  default=True,
                  help='Save code to W&B')
flags.DEFINE_integer(
    'debug_every',
    0,
    help='Run MUSDB debug visualization every N train steps (0 disables)',
)


class EMA(pl.Callback):

    def __init__(self, factor=.999) -> None:
        super().__init__()
        self.weights = {}
        self.factor = factor

    def on_train_batch_end(self, trainer, pl_module, outputs, batch,
                           batch_idx) -> None:
        for n, p in pl_module.named_parameters():
            if n not in self.weights:
                self.weights[n] = p.data.clone()
                continue

            self.weights[n] = self.weights[n] * self.factor + p.data * (
                1 - self.factor)

    def swap_weights(self, module):
        for n, p in module.named_parameters():
            current = p.data.clone()
            p.data.copy_(self.weights[n])
            self.weights[n] = current

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        if self.weights:
            self.swap_weights(pl_module)
        else:
            print("no ema weights available")

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if self.weights:
            self.swap_weights(pl_module)
        else:
            print("no ema weights available")

    def state_dict(self) -> Dict[str, Any]:
        return self.weights.copy()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.weights.update(state_dict)


class MusdbDebugVisualizerCallback(pl.Callback):

    DEBUG_SECONDS = 3.0

    def __init__(self, debug_every: int, fallback_val_loader: Optional[DataLoader] = None) -> None:
        super().__init__()
        self.debug_every = debug_every
        self._fallback_val_loader = fallback_val_loader
        self._cached_sample: Optional[torch.Tensor] = None
        self._viz_hook = None
        self._viz_class = None

    @staticmethod
    def _extract_tensor(sample):
        if isinstance(sample, torch.Tensor):
            return sample
        return torch.as_tensor(sample)

    @staticmethod
    def _resolve_single_dataloader(dataloaders):
        if isinstance(dataloaders, (list, tuple)):
            if not dataloaders:
                return None
            return dataloaders[0]
        return dataloaders

    def on_fit_start(self, trainer, pl_module) -> None:
        device = getattr(getattr(trainer, 'strategy', None), 'root_device', pl_module.device)
        if device.type != 'cuda':
            raise RuntimeError(
                'MUSDB debug visualizer requires CUDA tensors. '
                'Run training on CUDA or disable --debug_every.'
            )

        val_loader = self._resolve_single_dataloader(getattr(trainer, 'val_dataloaders', None))
        if val_loader is None:
            val_loader = self._resolve_single_dataloader(self._fallback_val_loader)
        if val_loader is None:
            raise RuntimeError(
                'MUSDB debug visualizer enabled but validation dataloader is unavailable at fit start.'
            )

        val_dataset = getattr(val_loader, 'dataset', None)
        if val_dataset is None or len(val_dataset) == 0:
            raise RuntimeError(
                'MUSDB debug visualizer enabled but validation dataset is empty.'
            )

        sample = self._extract_tensor(val_dataset[0]).detach().to(device)
        if sample.dim() == 2:
            sample = sample.unsqueeze(0)
        elif sample.dim() != 3:
            raise RuntimeError(
                f'Unsupported MUSDB debug sample shape {tuple(sample.shape)}; expected [C, T] or [B, C, T].'
            )

        sample_rate = int(getattr(pl_module, 'sr', 0))
        if sample_rate <= 0:
            raise RuntimeError('MUSDB debug visualizer requires a positive model sample rate.')

        target_samples = int(round(sample_rate * self.DEBUG_SECONDS))
        if sample.shape[-1] < target_samples:
            sample = F.pad(sample, (0, target_samples - sample.shape[-1]))
        else:
            sample = sample[..., :target_samples]

        self._cached_sample = sample

        try:
            from musicsep_visualizer import VisualizationHook
        except ImportError as err:
            raise RuntimeError(
                'MUSDB debug visualizer requested but musicsep-visualizer is not installed. '
                'Install dependencies from requirements.txt.'
            ) from err

        self._viz_class = VisualizationHook
        self._viz_hook = VisualizationHook('musdb_debug_output')

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if self._cached_sample is None or self._viz_hook is None:
            return

        if trainer.global_step <= 0 or trainer.global_step % self.debug_every:
            return

        was_training = pl_module.training
        pl_module.eval()

        try:
            with torch.no_grad():
                output = pl_module(self._cached_sample)
                self._viz_hook(output)
        finally:
            if was_training:
                pl_module.train()

    def on_fit_end(self, trainer, pl_module) -> None:
        if self._viz_hook is None or self._viz_class is None:
            return

        stop_visualization = getattr(self._viz_class, 'stop_visualization', None)
        if callable(stop_visualization):
            stop_visualization()

def add_gin_extension(config_name: str) -> str:
    if config_name[-4:] != '.gin':
        config_name += '.gin'
    return config_name

def parse_augmentations(augmentations):
    for a in augmentations:
        gin.parse_config_file(a)
        add_augmentation()
        gin.clear_config()
    return get_augmentations()


def resolve_musdb_roots(db_path: str, val_db_path: str | None):
    if val_db_path:
        train_root = db_path
        val_root = val_db_path
    else:
        train_root = os.path.join(db_path, 'train')
        val_root = os.path.join(db_path, 'test')

    if not os.path.isdir(train_root):
        raise RuntimeError(f"MUSDB train root does not exist: {train_root}")
    if not os.path.isdir(val_root):
        raise RuntimeError(f"MUSDB val root does not exist: {val_root}")
    return train_root, val_root


def infer_musdb_channels(split_root: str, stem_filename: str, target_channels: int):
    if target_channels > 0:
        return target_channels

    for root, _, names in os.walk(split_root):
        if stem_filename not in names:
            continue

        path = os.path.join(root, stem_filename)
        try:
            info = torchaudio.info(path)
        except Exception:
            continue

        if info.num_channels and info.num_channels > 0:
            return int(info.num_channels)

    print(
        f"[Warning] could not infer channels from {split_root} with stem '{stem_filename}', taking 1 by default"
    )
    return 1


def get_logger(run_name: str):
    if FLAGS.wandb:
        logger = pl.loggers.WandbLogger(
            name=run_name,
            project=FLAGS.wandb_project,
            entity=FLAGS.wandb_entity,
            save_dir=FLAGS.out_path,
            log_model=False,
        )

        if FLAGS.wandb_save_code:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            if not os.path.isdir(repo_root):
                repo_root = os.getcwd()
            logger.experiment.log_code(root=repo_root)

        return logger

    return pl.loggers.TensorBoardLogger(
        FLAGS.out_path,
        name=run_name,
    )

def main(argv):
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True

    # check dataset channels
    if FLAGS.dataset_format == 'lmdb':
        n_channels = rave.dataset.get_training_channels(FLAGS.db_path, FLAGS.channels)
    else:
        train_root, val_root = resolve_musdb_roots(FLAGS.db_path, FLAGS.val_db_path)
        n_channels = infer_musdb_channels(train_root, FLAGS.musdb_stem, FLAGS.channels)
        print(f'MUSDB train root: {train_root}')
        print(f'MUSDB val root: {val_root}')

    gin.bind_parameter('RAVE.n_channels', n_channels)

    # parse augmentations
    augmentations = parse_augmentations(map(add_gin_extension, FLAGS.augment))
    gin.bind_parameter('dataset.get_dataset.augmentations', augmentations)
    gin.bind_parameter('dataset.get_dataset_pair.augmentations', augmentations)

    # parse configuration
    if FLAGS.ckpt:
        config_file = rave.core.search_for_config(FLAGS.ckpt)
        if config_file is None:
            print('Config file not found in %s'%FLAGS.run)
        gin.parse_config_file(config_file)
    else:
        gin.parse_config_files_and_bindings(
            map(add_gin_extension, FLAGS.config),
            FLAGS.override,
        )

    # create model
    model = rave.RAVE(n_channels=n_channels)
    if FLAGS.derivative:
        model.integrator = rave.dataset.get_derivator_integrator(model.sr)[1]

    # parse datasset
    if FLAGS.dataset_format == 'lmdb':
        dataset = rave.dataset.get_dataset(FLAGS.db_path,
                                           model.sr,
                                           FLAGS.n_signal,
                                           derivative=FLAGS.derivative,
                                           normalize=FLAGS.normalize,
                                           rand_pitch=FLAGS.rand_pitch,
                                           n_channels=n_channels)
        train, val = rave.dataset.split_dataset(dataset, 98)
    else:
        train, val = rave.dataset.get_dataset_pair(
            train_root=train_root,
            val_root=val_root,
            sr=model.sr,
            n_signal=FLAGS.n_signal,
            stem_filename=FLAGS.musdb_stem,
            derivative=FLAGS.derivative,
            normalize=FLAGS.normalize,
            rand_pitch=FLAGS.rand_pitch,
            augmentations=augmentations,
            n_channels=n_channels,
        )

    # get data-loader
    num_workers = FLAGS.workers
    if os.name == "nt" or sys.platform == "darwin":
        num_workers = 0
    train = DataLoader(train,
                       FLAGS.batch,
                       True,
                       drop_last=True,
                       num_workers=num_workers)
    val = DataLoader(val, FLAGS.batch, False, num_workers=num_workers)

    # CHECKPOINT CALLBACKS
    validation_checkpoint = pl.callbacks.ModelCheckpoint(monitor="validation",
                                                         filename="best")
    last_filename = "last" if FLAGS.save_every is None else "epoch-{epoch:04d}"                                                        
    last_checkpoint = rave.core.ModelCheckpoint(filename=last_filename, step_period=FLAGS.save_every)

    val_check = {}
    if len(train) >= FLAGS.val_every:
        val_check["val_check_interval"] = 1 if FLAGS.smoke_test else FLAGS.val_every
    else:
        nepoch = FLAGS.val_every // len(train)
        val_check["check_val_every_n_epoch"] = nepoch

    if FLAGS.smoke_test:
        val_check['limit_train_batches'] = 1
        val_check['limit_val_batches'] = 1

    gin_hash = hashlib.md5(
        gin.operative_config_str().encode()).hexdigest()[:10]

    RUN_NAME = f'{FLAGS.name}_{gin_hash}'

    os.makedirs(os.path.join(FLAGS.out_path, RUN_NAME), exist_ok=True)

    if FLAGS.gpu == [-1]:
        gpu = 0
    else:
        gpu = FLAGS.gpu or rave.core.setup_gpu()

    print('selected gpu:', gpu)

    accelerator = None
    devices = None
    if FLAGS.gpu == [-1]:
        pass
    elif torch.cuda.is_available():
        accelerator = "cuda"
        devices = FLAGS.gpu or rave.core.setup_gpu()
    elif torch.backends.mps.is_available():
        print(
            "Training on mac is not available yet. Use --gpu -1 to train on CPU (not recommended)."
        )
        exit()
        accelerator = "mps"
        devices = 1

    callbacks = [
        validation_checkpoint,
        last_checkpoint,
        rave.model.WarmupCallback(),
        rave.model.QuantizeCallback(),
        # rave.core.LoggerCallback(rave.core.ProgressLogger(RUN_NAME)),
        rave.model.BetaWarmupCallback(),
    ]

    if FLAGS.ema is not None:
        callbacks.append(EMA(FLAGS.ema))

    if FLAGS.debug_every > 0:
        if FLAGS.dataset_format == 'musdb':
            callbacks.append(MusdbDebugVisualizerCallback(FLAGS.debug_every, fallback_val_loader=val))
            print(
                f'MUSDB debug visualizer enabled: running every {FLAGS.debug_every} train steps.'
            )
        else:
            print(
                '[Warning] --debug_every is only supported with --dataset_format musdb. '
                'Debug visualizer is disabled.'
            )

    trainer = pl.Trainer(
        logger=get_logger(RUN_NAME),
        accelerator=accelerator,
        devices=devices,
        callbacks=callbacks,
        max_epochs=300000,
        max_steps=FLAGS.max_steps,
        profiler="simple",
        enable_progress_bar=FLAGS.progress,
        **val_check,
    )

    run = rave.core.search_for_run(FLAGS.ckpt)
    if run is not None:
        print('loading state from file %s'%run)

    with open(os.path.join(FLAGS.out_path, RUN_NAME, "config.gin"), "w") as config_out:
        config_out.write(gin.operative_config_str())

    trainer.fit(model, train, val, ckpt_path=run)


if __name__ == "__main__": 
    app.run(main)
