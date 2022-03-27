import argparse
from loguru import logger
import numpy as np
import os
from config import source_env, target_env, data_mode, backend, backends, snr
from utils.torch_utils import generate_save_predictions
from utils.evaluate_utils import compute_sed_f1_errorrate
from utils.data_utils import data_modes
from utils.pl_utils import LM


@logger.catch
def evaluate(args):
    source_env = args.source_env
    target_env = args.target_env
    data_mode = args.data_mode
    expt_name = args.expt_name

    logger.info(
        f'Starting evalution on {data_mode} data for source - {source_env} and target - {target_env}.')
    logger.add(
        f'{args.backend}_eval_src_{source_env}_target_{target_env}.log', rotation='500 KB')
    logger.info(f'Loading best f1 model for {source_env} audio.')
    model_ckpt_folder_path = os.path.join(os.path.dirname(__file__),
                                          'model_checkpoints', f'{snr}-mono', backends[0], expt_name)
    chkpt_path = os.path.join(model_ckpt_folder_path,
                              f"model-{source_env}-best-f1.ckpt")
    if not os.path.exists:
        raise Exception(f'Not found: {chkpt_path}.')
    model = LM.load_from_checkpoint(chkpt_path)
    logger.info(f'Loaded model checkpoint: {chkpt_path}')
    reference_files, estimated_files = generate_save_predictions(
        model, data_mode, target_env)
    curr_f1, curr_error = compute_sed_f1_errorrate(
        reference_files, estimated_files)

    logger.success("F-measure: {:.3f}".format(curr_f1))
    logger.success("Error rate: {:.3f}".format(curr_error))

    # Or print all metrics as reports
    return (curr_f1, curr_error)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='For making realtime predictons.')
    parser.add_argument('-en', '--expt_name', type=str, required=True)
    parser.add_argument('-b', '--backend', type=str,
                        default=backend, choices=backends)
    parser.add_argument('-se', '--source_env', type=str, default=source_env)
    parser.add_argument('-te', '--target_env', type=str, default=target_env)
    parser.add_argument('-m', '--data_mode', type=str,
                        default=data_mode, choices=data_modes)

    args = parser.parse_args()
    f1, error = evaluate(args)
