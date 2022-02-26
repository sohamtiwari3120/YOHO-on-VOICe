from typing import List
import sed_eval
import dcase_util


def compute_sed_f1_errorrate(reference_file_paths: List[str], estimated_file_paths: List[str]):
    """Function to compute f1 and error rate for sound event detection using sed_eval.

    Args:
        reference_file_paths (List[str]): List of paths to ground truth annotations.
        estimated_file_paths (List[str]): List of paths to model predicted annotations.

    Returns:
        float: F1-score for SED
        float: Error rate for SED
    """    
    file_list = []
    for reference, estimated in zip(reference_file_paths, estimated_file_paths):
        file_list.append({
            'reference_file': reference,
            'estimated_file': estimated
        })

    data = []
    all_data = dcase_util.containers.MetaDataContainer()
    for file_pair in file_list:
        reference_event_list = sed_eval.io.load_event_list(
            filename=file_pair['reference_file']
        )
        estimated_event_list = sed_eval.io.load_event_list(
            filename=file_pair['estimated_file']
        )
        data.append({'reference_event_list': reference_event_list,
                    'estimated_event_list': estimated_event_list})  
        all_data += reference_event_list
    event_labels = all_data.unique_event_labels 

    segment_based_metrics = sed_eval.sound_event.SegmentBasedMetrics(
        event_label_list=event_labels,
        time_resolution=1.0
    )   
    event_based_metrics = sed_eval.sound_event.EventBasedMetrics(
        event_label_list=event_labels,
        t_collar=1.0
    )

    for file_pair in data:
        segment_based_metrics.evaluate(
            reference_event_list=file_pair['reference_event_list'],
            estimated_event_list=file_pair['estimated_event_list']
        )   
        event_based_metrics.evaluate(
            reference_event_list=file_pair['reference_event_list'],
            estimated_event_list=file_pair['estimated_event_list']
        )   

    overall_segment_based_metrics = segment_based_metrics.results_overall_metrics()
    curr_f1 = overall_segment_based_metrics['f_measure']['f_measure']
    curr_error = overall_segment_based_metrics['error_rate']['error_rate']
    
    return curr_f1, curr_error