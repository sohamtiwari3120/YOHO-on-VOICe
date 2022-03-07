import argparse
from models.YOHO import YohoModel
from utils.data_utils import VOICeDataModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from utils.torch_utils import MonitorSedF1Callback
from loguru import logger
from config import env, devices, accelerator, gradient_clip_val, input_height, input_width
from torchsummary import summary
import torch

@logger.catch
def main(args):
    env = args.env
    logger.add(f'train_{env}.log', rotation='500 KB')
    logger.info(f'Starting training of model for {env} audio.')
    if args.chkpt_path is not None:
        model = YohoModel.load_from_checkpoint(args.chkpt_path)
        logger.info(f'Loaded model checkpoint: {args.chkpt_path}')
    else:
        model = YohoModel()
        logger.info(f'Starting a fresh model.')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary(model.to(device), (1, input_height, input_width))
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = Trainer(callbacks=[MonitorSedF1Callback(env), lr_monitor], devices=devices, accelerator=accelerator, gradient_clip_val=gradient_clip_val)
    voice_dm = VOICeDataModule(env)

    if args.auto_lr:
        logger.info(f'Starting auto lr find of model for {env} audio.')
        lr_finder = trainer.tuner.lr_find(model, voice_dm)
        logger.info(f'Finished auto lr find of model for {env} audio.')
        # Results can be found in
        lr_finder.results
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='For making realtime predictons.')
    parser.add_argument('-e', '--env', type=str, default=env)
    parser.add_argument('-cp', '--chkpt_path', type=str, default=None)
    parser.add_argument('-alr', '--auto_lr', type=bool, default=False)

    args = parser.parse_args()
    main(args)
