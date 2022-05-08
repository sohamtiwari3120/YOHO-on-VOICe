import argparse
from torchsummary import summary
import torch
import os
from datetime import datetime
from loguru import logger
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
from config import hparams, YOHO_hparams
hp = hparams()
seed = hp.seed

seed_everything(seed, workers=True)

from utils.torch_utils import MonitorSedF1Callback  # NOQA
from utils.tf_utils import DataGenerator, my_loss_fn  # NOQA
from utils.pl_utils import LM  # NOQA
from models.YOHO import Yoho  # NOQA
from models.YOHO_tf import YohoTF  # NOQA
from models.VOICeConvNeXt import VOICeConvNeXt  # NOQA
from models.pann_encoder import VOICePANN, VOICePANNYoho  # NOQA
from models.ViT import VOICeViT  # NOQA
from models.VOICeCoAtNet import VOICeCoAtNet  # NOQA
from models.VOICeConvMixer import VOICeConvMixer  # NOQA
from utils.data_utils import VOICeDataModule  # NOQA


@logger.catch
def pytorch(args):
    env = args.env
    expt_name = args.expt_name
    date_today = datetime.today().strftime('%d%m%Y')
    expt_folder = os.path.join(os.path.dirname(__file__),
                               'model_checkpoints', f'{hp.snr}-mono', f'{args.backend}', date_today, expt_name)
    if not os.path.exists(expt_folder):
        os.makedirs(expt_folder)

    logger.add(os.path.join(
        expt_folder, f'{args.backend}_train_{env}.log'))
    logger.info(f'Using {args.backend} backend')
    logger.info(
        f'{expt_name}: Starting training of model for {env} audio. Saving checkpoints and logs in {expt_folder}')

    lightning_model_class = eval(args.model_name)

    if args.chkpt_path is not None:
        model = lightning_model_class.load_from_checkpoint(args.chkpt_path)
        logger.info(f'Loaded model checkpoint: {args.chkpt_path}')
    else:
        if args.model_name == "Yoho":
            model = lightning_model_class(use_cbam=args.use_cbam, use_pna=args.use_pna, use_ufo=args.use_ufo, use_mva=args.use_mva,
                       use_mish_activation=args.use_mish_activation, use_serf_activation=args.use_serf_activation, use_patches=args.use_patches, use_residual=args.use_residual, use_rectangular=args.use_rectangular)
        elif args.model_name == "VOICeCoAtNet":
            model = lightning_model_class(use_cbam=args.use_cbam)
        else:
            model = LM(eval(args.model_name)(use_cbam=args.use_cbam))
        logger.info(f'Starting a fresh model.')
        logger.info(f'use_cbam = {args.use_cbam}')

    params = hp.__dict__
    params.update(vars(args))
    logger.info(params)

    if args.no_wandb:
        tb_logger = pl_loggers.TensorBoardLogger(os.path.join(
            os.path.dirname(__file__), 'lightning_logs', date_today, expt_name))
    else:
        wandb_logger = pl_loggers.WandbLogger(
            project='YOHO-on-VOICe', name=f"{date_today}/{expt_name}")
        wandb_logger.experiment.config.update(
            hp.__dict__, allow_val_change=True)
        wandb_logger.experiment.config.update(args, allow_val_change=True)
        wandb_logger.watch(model)

    if args.model_summary:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        summary(model.to(device), (1, hp.input_height, hp.input_width))

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    earlystopping = EarlyStopping(
        monitor=hp.es_monitor, mode=hp.es_mode, patience=hp.es_patience)

    torch.use_deterministic_algorithms(not args.indeterministic)
    trainer = Trainer(callbacks=[MonitorSedF1Callback(
        env, expt_folder), lr_monitor, earlystopping], devices=hp.devices, accelerator=hp.accelerator, gradient_clip_val=hp.gradient_clip_val, logger=tb_logger if args.no_wandb else wandb_logger, profiler='simple', deterministic=not args.indeterministic)
    voice_dm = VOICeDataModule(env)

    if args.auto_lr:
        logger.info(f'Starting auto lr find of model for {env} audio.')
        lr_finder = trainer.tuner.lr_find(model, voice_dm)
        logger.info(f'Finished auto lr find of model for {env} audio.')
        # Results can be found in lr_finder.results
        # Plot with
        fig = lr_finder.plot(suggest=True)
        fig.savefig('lr_finder.plot.png')
        # Pick point based on plot, or get suggestion
        new_lr = lr_finder.suggestion()
        logger.info(f'new_lr = {new_lr}')
        # update hparams of the model
        model.learning_rate = new_lr
        model.hparams.learning_rate = new_lr

    # Fit model
    trainer.fit(model, voice_dm)
    logger.info(f'Finished training of model for {env} audio.')


@logger.catch
def tensor_flow(args):
    env = args.env
    expt_name = args.expt_name
    date_today = datetime.today().strftime('%d%m%Y')
    expt_folder = os.path.join(os.path.dirname(__file__),
                               'model_checkpoints', f'{hp.snr}-mono', f'{args.backend}', date_today, expt_name)
    if not os.path.exists(expt_folder):
        os.makedirs(expt_folder)

    logger.add(os.path.join(
        expt_folder, f'{args.backend}_train_{env}.log'))
    logger.info(f'Using {args.backend} backend')
    logger.info(f'{expt_name}: Starting training of model for {env} audio.')
    model = YohoTF(expt_folder, env)
    if args.chkpt_path is not None:
        model.load_from_checkpoint(args.chkpt_path)
        logger.info(f'Loaded model checkpoint: {args.chkpt_path}')
    else:
        logger.info(f'Starting a fresh model.')
    if args.model_summary:
        model.summary()

    train_data = DataGenerator('training', env, spec_transform=True)
    validation_data = DataGenerator(
        'validation', env, spec_transform=False, shuffle=False)

    # Fit model
    model.train_and_validate(train_data, validation_data, my_loss_fn)
    logger.info(f'Finished training of model for {env} audio.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='For making realtime predictons.')
    parser.add_argument('-en', '--expt_name', type=str, required=True)
    parser.add_argument('-b', '--backend', type=str,
                        default=hp.backend, choices=hp.backends)
    parser.add_argument('-m', '--model_name', type=str,
                        default=hp.model_name, choices=hp.models)
    parser.add_argument('-e', '--env', type=str, default=hp.env)
    parser.add_argument('-cp', '--chkpt_path', type=str, default=None)
    parser.add_argument('-alr', '--auto_lr', action='store_true')
    parser.add_argument('-ms', '--model_summary', action='store_true')
    parser.add_argument('-cbam', '--use_cbam', action='store_true')
    parser.add_argument('-pna', '--use_pna', action='store_true')
    parser.add_argument('-ufo', '--use_ufo', action='store_true')
    parser.add_argument('-mva', '--use_mva', action='store_true')
    parser.add_argument('-mish', '--use_mish_activation', action='store_true')
    parser.add_argument('-serf', '--use_serf_activation', action='store_true')
    parser.add_argument('-patchify', '--use_patches', action='store_true')
    parser.add_argument('-res', '--use_residual', action='store_true')
    parser.add_argument('-nwb', '--no_wandb', action="store_true")
    parser.add_argument('-indt', '--indeterministic', action="store_true")
    parser.add_argument('-rect', '--use_rectangular', action="store_true")

    args = parser.parse_args()
    eval(args.backend)(args)
