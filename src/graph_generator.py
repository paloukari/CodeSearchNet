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
    --pickle                     The input and output are pickle files. [default: False]

Example:

    python graph_generator.py \
    --azure-info /ds/hamel/azure_auth.json \
    ../resources/data/python/final/jsonl/train  \
    ../resources/data/python/final_graphs/jsonl/train

"""

import pickle
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
from ast_graph_generator import AstGraphGenerator, NODE_TYPE

from multiprocessing import cpu_count

def jsonl_to_df(input_folder: RichPath) -> pd.DataFrame:
    "Concatenates all jsonl files from path and returns them as a single pandas.DataFrame ."

    assert input_folder.is_dir(), 'Argument supplied must be a directory'
    dfs = []
    files = list(input_folder.iterate_filtered_files_in_dir('*.jsonl.gz'))
    assert files, 'There were no jsonl.gz files in the specified directory.'
    print(f'reading files from {input_folder.path}')
    for f in tqdm(files, total=len(files)):
        dfs.append(pd.DataFrame(list(f.read_as_jsonl(
            error_handling=lambda m, e: print(f'Error while loading {m} : {e}')))))
    return pd.concat(dfs), len(files)


def generate_graph_column(df: pd.DataFrame, is_pickle: bool) -> pd.DataFrame:
    
    if not is_pickle:
        assert 'code_tokens' in df.columns.values, 'Data must contain field code_tokens'
        assert 'language' in df.columns.values, 'Data must contain field language'

    def next_terminal(node, non_terminal):
        if node in non_terminal:
            return next_terminal(node+1, non_terminal)
        return node
    
    def fix_index(node, non_terminal):
        node = next_terminal(node, non_terminal)
        return node - len([_ for _ in non_terminal if _ < node])

    def generate_graph(code):
        try:
            if code.startswith('def get_taxon_to_species_dict():'):
                return None, None
            visitor = AstGraphGenerator()
            visitor.visit(parse(code))

            # we'll keep the the first node and rename it to root
            visitor.node_label[0] = 'root'
            # the non-terminal nodes
            n_t = [index for (index, _) in sorted(visitor.node_label.items(
            )) if visitor.node_type[index] == NODE_TYPE['non_terminal'] and index > 0]
            n_t.sort()

            E = [(t, origin, destination)
                    for (origin, destination), edges
                    in visitor.graph.items() for t in edges]
            
            # we'll replace the non terminal nodes and edges with the first terminal child
            E = [(e[0], fix_index(e[1], n_t), fix_index(e[2], n_t))
                    for e in E]
            # remove self references
            E = [e for e in E if e[1] != e[2]]

            V = [label.strip() for (index, label) in sorted(
                visitor.node_label.items()) if index not in n_t]
            return V, E
        except Exception as ex:
            # print(ex)
            return None, None

    tqdm.pandas()
    if not is_pickle:
        df['graph_V'], df['graph_E'] = zip(
            *df['original_string'].progress_map(generate_graph))
    else:
        df['graph_V'], df['graph_E'] = zip(
            *df['function'].progress_map(generate_graph))
    return df


def run(args):
    from_pickle = args.get('--pickle', False)
    
    if not from_pickle:
        azure_info_path = args.get('--azure-info', None)
        
        input_path = RichPath.create(args['INPUT_FILENAME'], azure_info_path)
        output_folder = args['OUTPUT_FOLDER']

        os.makedirs(output_folder, exist_ok=True)

        # get data and process it
        df, files = jsonl_to_df(input_path)
        print('Generating graphs ... this may take some time.')
        df = generate_graph_column(df, False)
        original_size = df.shape[0]
        df.dropna(subset=['graph_V', 'graph_E'], inplace=True)
        print(f"dropped {original_size - df.shape[0]} records")
        # save dataframes as chunked jsonl files

        print(f'Saving data to {str(output_folder)}')

        chunked_save_df_to_jsonl(df, RichPath.create(
            args['OUTPUT_FOLDER'], azure_info_path), files, False)
    
    else:
        input_path = args['INPUT_FILENAME']
        output_path = args['OUTPUT_FOLDER']

        definitions = pickle.load(open(input_path, 'rb'))
        number_of_splits = 8*cpu_count()
        definition_splits = np.array_split(definitions, number_of_splits)

        print('Generating graphs ... this may take some time.')

        for i, definition_split in enumerate(definition_splits):
            output_split_path = f'{output_path}_{i}'
            if os.path.exists(output_split_path):
                print(f'Skipping {output_split_path}..')
                continue

            df = pd.DataFrame(list(definition_split))
            original_size = df.shape[0]
            
            df = generate_graph_column(df, True)
            #df.dropna(subset=['graph_V', 'graph_E'], inplace=True)
            
            #print(f"Failed to create graphs for {original_size - df.shape[0]} records")
            definitions = df.to_dict('records')

            print(f'Saving data to {output_split_path}')
            with open(output_split_path, 'wb') as f:
                pickle.dump(definitions, f)
        
        definitions = []
        for i in range(number_of_splits):
            output_split_path = f'{output_path}_{i}'
            definitions = definitions + pickle.load(open(output_split_path, 'rb'))
        with open(output_path, 'wb') as f:
            pickle.dump(definitions, f)



if __name__ == '__main__':
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), args.get('--debug'))
