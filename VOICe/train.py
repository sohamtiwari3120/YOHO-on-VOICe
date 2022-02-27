import argparse
from models.YOHO import YohoModel
from utils.data_utils import VOICeDataModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from utils.torch_utils import MonitorSedF1Callback
from loguru import logger
from config import env

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
    trainer = Trainer(callbacks=[MonitorSedF1Callback(env), LearningRateMonitor], devices="auto", accelerator="auto")
    voice_dm = VOICeDataModule(env)
    trainer.fit(model, voice_dm)
    logger.info(f'Finished training of model for {env} audio.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='For making realtime predictons.')
    parser.add_argument('-e', '--env', type=str, default=env)
    parser.add_argument('-cp', '--chkpt_path', type=str, default=None)

    args = parser.parse_args()
    main(args)
