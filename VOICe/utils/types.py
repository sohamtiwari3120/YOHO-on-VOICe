from typing import List, Tuple, Dict

depthwise_layers_type = List[Tuple[Tuple[int, int], int, int]]
data_path_dictionary_types = Dict[str, List[str]]
file_paths_type = Dict[str, data_path_dictionary_types]