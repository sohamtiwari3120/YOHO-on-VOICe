import argparse
from loguru import logger
from config import hparams
from utils.data_utils import generate_windows_and_anns, save_logmelspec_and_labels, convert_to_mono, envs, data_modes
logger.add('process_dataset.log', rotation='500 KB')

hp = hparams()

@logger.catch
def main(args):
    logger.info(f'Starting processing of entire dataset for {hp.snr} audio.')
    if not args.skip_mono_conversion:
        logger.info(f'First converting all audios to mono.')
        convert_to_mono()
        logger.info(f'Finished conversion to mono audio.')
    logger.info(f"use_leaf = {hp.use_leaf}")
    for mode in args.data_modes:
        for env in args.envs:
            logger.info(f'Processing {mode}, {env} data.')
            audio_windows, labels = generate_windows_and_anns(mode, env)
            print("len(audio_windows)", len(audio_windows))
            print("len(labels)", len(labels))
            print("hp.save_labels", hp.save_labels)
            print("hp.save_logmelspec", hp.save_logmelspec)
            print("hp.sample_rate", hp.sample_rate)
            logmel_path, label_path = save_logmelspec_and_labels(mode, env, audio_windows, labels)
            logger.info(f'For mode:{mode}, env:{env}')
            logger.info(f'Saved spectrograms in path: {logmel_path}')
            logger.info(f'Saved labels in path: {label_path}')
    logger.info(f'Finished processing entire dataset.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Program to process dataset by converting to mono audios and then generating and saving spectrograms.')
    parser.add_argument('-smc', '--skip_mono_conversion', action='store_true')
    parser.add_argument('-dms', '--data_modes', nargs='+', default=data_modes)
    parser.add_argument('-es', '--envs', nargs='+', default=envs)
    args = parser.parse_args()
    main(args)