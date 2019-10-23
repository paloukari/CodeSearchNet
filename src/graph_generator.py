#!/usr/bin/env python
"""
Generate the graph data from the jsonl.gz files.

Usage:
    graph_generator.py [options] INPUT_FILENAME OUTPUT_FOLDER

Arguments:
    INPUT_FOLDER               directory w/ compressed jsonl files that have a .jsonl.gz a file extension
    OUTPUT_FOLDER              directory where you want to save data to.

Options:
    -h --help                    Show this screen.
    --debug                      Enable debug routines. [default: False]

Example:

    python graph_generator.py \
    --azure-info /ds/hamel/azure_auth.json \
    ../resources/data/python/final/jsonl/train  \
    ../resources/data/python/final_graphs/jsonl/train

"""

from docopt import docopt
import hashlib
import pandas as pd
from utils.pkldf2jsonl import chunked_save_df_to_jsonl
from dpu_utils.utils import RichPath, run_and_debug
import os
from tqdm import tqdm
from typing import List, Dict, Any, Iterable, Tuple, Optional, Union, Callable, Type, DefaultDict
import numpy as np

from ast import parse
from ast_graph_generator import AstGraphGenerator

def jsonl_to_df(input_folder: RichPath) -> pd.DataFrame:
    "Concatenates all jsonl files from path and returns them as a single pandas.DataFrame ."

    assert input_folder.is_dir(), 'Argument supplied must be a directory'
    dfs = []
    files = list(input_folder.iterate_filtered_files_in_dir('*.jsonl.gz'))
    assert files, 'There were no jsonl.gz files in the specified directory.'
    print(f'reading files from {input_folder.path}')
    for f in tqdm(files, total=len(files)):
        dfs.append(pd.DataFrame(list(f.read_as_jsonl(error_handling=lambda m,e: print(f'Error while loading {m} : {e}')))))
    return pd.concat(dfs), len(files)


def generate_graph_column(df: pd.DataFrame) -> pd.DataFrame:
    assert 'code_tokens' in df.columns.values, 'Data must contain field code_tokens'
    assert 'language' in df.columns.values, 'Data must contain field language'
    
    def generate_graph(code):
        try:
            visitor = AstGraphGenerator()
            visitor.visit(parse(code))
            E = [(t, origin, destination)
                                    for (origin, destination), edges
                                    in visitor.graph.items() for t in edges]
            V = [label.strip() for (_, label) in sorted(visitor.node_label.items())]
            return V, E
        except Exception as ex:
            #print(ex)
            return None,None
    
    tqdm.pandas()
    df['graph_V'], df['graph_E'] = zip(*df['original_string'].progress_map(generate_graph))
    #df['graph'] = df.apply(lambda x: generate_graph(visitor, x['original_string']), axis=1)
    return df

def run(args):

    azure_info_path = args.get('--azure-info', None)
    input_path = RichPath.create(args['INPUT_FILENAME'], azure_info_path)
    output_folder = args['OUTPUT_FOLDER']
    
    os.makedirs(output_folder, exist_ok=True)
    
    # get data and process it
    df, files = jsonl_to_df(input_path)
    print('Generating graphs ... this may take some time.')
    df = generate_graph_column(df)
    original_size = df.shape[0]
    df.dropna(subset=['graph_V', 'graph_E'], inplace=True)
    print(f"dropped {original_size - df.shape[0]} records")
    # save dataframes as chunked jsonl files
    
    
    print(f'Saving data to {str(output_folder)}')
    
    chunked_save_df_to_jsonl(df, RichPath.create(args['OUTPUT_FOLDER'], azure_info_path), files, False)

if __name__ == '__main__':
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), args.get('--debug'))
