from loguru import logger
from config import snr
from utils.data_utils import generate_windows_and_anns, save_logmelspec_and_labels, convert_to_mono, envs, data_modes
logger.add('process_dataset.log', rotation='500 KB')

@logger.catch
def main():
    logger.info(f'Starting processing of entire dataset for {snr} audio.')
    logger.info(f'First converting all audios to mono.')
    convert_to_mono()
    logger.info(f'Finished conversion to mono audio.')
    for mode in data_modes:
        for env in envs:
            logger.info(f'Processing {mode}, {env} data.')
            audio_windows, labels = generate_windows_and_anns(mode, env)
            logmel_path, label_path = save_logmelspec_and_labels(mode, env, audio_windows, labels)
            logger.info(f'For mode:{mode}, env:{env}')
            logger.info(f'Saved spectrograms in path: {logmel_path}')
            logger.info(f'Saved labels in path: {label_path}')
    logger.info(f'Finished processing entire dataset.')

if __name__ == '__main__':
    main()