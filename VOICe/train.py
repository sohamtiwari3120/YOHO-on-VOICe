import argparse
from models.YOHO import Yoho
from models.YOHO_tf import YohoTF
from models.VOICeConvNeXt import VOICeConvNeXt
from models.pann_encoder import VOICePANN, VOICePANNYoho
from models.ViT import VOICeViT
from models.VOICeCoAtNet import VOICeCoAtNet
from models.VOICeConvMixer import VOICeConvMixer
from utils.data_utils import VOICeDataModule
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor
from utils.torch_utils import MonitorSedF1Callback
from utils.tf_utils import DataGenerator, my_loss_fn
from utils.pl_utils import LM
from loguru import logger
from config import hparams
from torchsummary import summary
import torch
import os
from datetime import datetime

hp = hparams()


@logger.catch
def pytorch(args):
    env = args.env
    expt_name = args.expt_name
    date_today = datetime.today().strftime('%d%m%Y')
    # tb_logger = pl_loggers.TensorBoardLogger(os.path.join(
    #     os.path.dirname(__file__), 'lightning_logs', date_today, expt_name))
    wandb_logger = pl_loggers.WandbLogger(project='YOHO-on-VOICe', name=f"{date_today}/{expt_name}")
    wandb_logger.experiment.config.update(args)
    expt_folder = os.path.join(os.path.dirname(__file__),
                               'model_checkpoints', f'{hp.snr}-mono', f'{args.backend}', date_today, expt_name)
    if not os.path.exists(expt_folder):
        os.makedirs(expt_folder)

    logger.add(os.path.join(
        expt_folder, f'{args.backend}_train_{env}.log'))
    logger.info(f'Using {args.backend} backend')
    logger.info(
        f'{expt_name}: Starting training of model for {env} audio. Saving checkpoints and logs in {expt_folder}')

    if args.chkpt_path is not None:
        model = LM.load_from_checkpoint(args.chkpt_path)
        logger.info(f'Loaded model checkpoint: {args.chkpt_path}')
    else:
        
        model = LM(eval(args.model_name)(use_cbam=args.use_cbam))
        logger.info(f'Starting a fresh model.')
        logger.info(f'use_cbam = {args.use_cbam}')

    if args.model_summary:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        summary(model.to(device), (1, hp.input_height, hp.input_width))

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = Trainer(callbacks=[MonitorSedF1Callback(
        env, expt_folder), lr_monitor], devices=hp.devices, accelerator=hp.accelerator, gradient_clip_val=hp.gradient_clip_val, logger=wandb_logger, profiler='simple')
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

    args = parser.parse_args()
    eval(args.backend)(args)
