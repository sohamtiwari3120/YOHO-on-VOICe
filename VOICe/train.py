import argparse
from models.YOHO import YohoModel
from utils.data_utils import VOICeDataModule
from pytorch_lightning import Trainer
from utils.torch_utils import MonitorSedF1Callback
from loguru import logger

@logger.catch
def main(args):
    env = args.env
    logger.add(f'train_{env}.log', rotation='500 KB')
    logger.info(f'Starting training of model for {env} audio.')
    model = YohoModel()
    trainer = Trainer(callbacks=[MonitorSedF1Callback(env)], devices="auto", accelerator="auto")
    indoor_voice_dm = VOICeDataModule(env)
    trainer.fit(model, indoor_voice_dm)
    logger.info(f'Finished training of model for {env} audio.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='For making realtime predictons.')
    parser.add_argument('-e', '--env', type=str, default='indoor')

    args = parser.parse_args()
    main(args)
