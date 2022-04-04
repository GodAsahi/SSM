# coding:utf-8

import argparse
from collections import OrderedDict
import os
import time
import numpy as np
from rdkit import Chem
import torch
from utils import enumerate_molecule
from utils import usrcat,get_properties
from torch.autograd import Variable
import torch.multiprocessing as mp
from torch.multiprocessing import Pool
import torch.nn as nn
import torch.optim as optim

from model import SSM
from opt import SharedRMSprop, SharedAdam
import utils

def normalize(v, max_v, min_v):
    v = min(max_v, v)
    v = max(min_v, v)
    return (v - min_v) / (max_v - min_v)


def sample(shared_model, wholes, scaffolds, condition1, condition2, pid, retval_list, args):
    model = SSM(args)
    st1 = time.time()

    for idx, (s1, s2) in enumerate(zip(wholes, scaffolds)):
        model.load_state_dict(shared_model.state_dict())
        retval = model.sample(s1, s2, latent_vector=None, condition1=condition1, condition2=condition2,
                              stochastic=args.stochastic)
        if retval is None: continue
        retval_list[pid].append((s1, retval))
    end1 = time.time()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ncpus', help='number of cpus', type=int, default=2)
    parser.add_argument('--item_per_cycle', help='number of generations per CPU', type=int, default=128)
    parser.add_argument('--dim_of_node_vector', help='dimension of node_vector', type=int, default=128)
    parser.add_argument('--dim_of_edge_vector', help='dimension of edge vector', type=int, default=128)
    parser.add_argument('--dim_of_FC', help='dimension of FC', type=int, default=128)

    parser.add_argument('--smiles_path', help='SMILES-data path', default='')
    parser.add_argument('--usrcat', help='scaffold generalization function', type=bool, default=True)
    parser.add_argument('--usrcat_threshold', help='threshold of usrcat', type=float, default=0)

    parser.add_argument('--save_fpath', help='file path of saved model', type=str, default='')
    parser.add_argument('--scaffold', help='smiles of scaffold', type=str, default='')
    parser.add_argument('--target_properties', help='values of target properties', nargs='+', default=[],
                        type=float)
    parser.add_argument('--scaffold_properties', help='values of scaffold properties[MW Logp tPSA]', nargs='+',
                        default=[], type=float)
    parser.add_argument('--output_filename', help='output file name', type=str, default='')
    parser.add_argument('--minimum_values', help='minimum values of properties. It will be used for normalization',
                        nargs='+', default=[], type=float)
    parser.add_argument('--maximum_values', help='maximum values of properties. It will be used for normalization',
                        nargs='+', default=[], type=float)
    parser.add_argument('--chiral', help='Should the chiral center be consistent with the scaffold', type=bool,
                        default=False)
    parser.add_argument('--stochastic', help='stocahstically add node and edge', action='store_true')
    args = parser.parse_args()


    save_fpath = os.path.realpath(os.path.expanduser(args.save_fpath))

    if args.usrcat==True:
        smiles_path=os.path.realpath(os.path.expanduser(args.smiles_path))
        l = []
        sm_sc = []
        with open(smiles_path, 'r') as f_r:
            content = f_r.read().split('\n')
            for i in content:
                i = i.split('\t')
                l.append(i)
        for i in l:
            sm_sc.append(i[1])
            sm_sc.append(i[2])

        re_smiles=usrcat(args.scaffold,sm_sc,args.usrcat_threshold)

        n=0
        for s in re_smiles:
            n=n+1
            ra_of_pr='{}/{}'.format(n,len(re_smiles))
            args.scaffold=s

            proper = []
            c = get_properties(True,s)
            for i in c:
                proper.append(float('%.3f' % i))

            args.scaffold_properties=proper

            output_filename = os.path.realpath(os.path.expanduser(str(n) + '.txt'))
            args.output_filename=output_filename

            target_properties = [normalize(*values) for values in
                                 zip(args.target_properties, args.maximum_values, args.minimum_values)]
            scaffold_properties = [normalize(*values) for values in
                                   zip(args.scaffold_properties, args.maximum_values, args.minimum_values)]

            mp.set_start_method('spawn',force=True)
            torch.manual_seed(1)

            args.N_conditions = len(target_properties) + len(scaffold_properties)

            shared_model = SSM(args)
            shared_model.share_memory()

            print("Model #Params: %dK" % (sum([x.nelement() for x in shared_model.parameters()]) / 1000,))
            print(f"""\
            Rate_of_progress  : {ra_of_pr}
            Ncpus             : {args.ncpus}
            Output_filename   : {args.output_filename }
            OMP_NUM_THREADS   : {os.environ.get('OMP_NUM_THREADS')}
            Num of generations: {args.item_per_cycle} per CPU (Total {args.ncpus * args.item_per_cycle})
            Chiral            : {args.chiral}
            Smiles_path       : {args.smiles_path}
            Usrcat            : {args.usrcat}
            Usrcat_threshold  : {args.usrcat_threshold}
            Model path        : {save_fpath}
            Output path       : {output_filename}
            Scaffold          : {args.scaffold}
            Scaffold values   : {args.scaffold_properties} -> {scaffold_properties}
            Target values     : {args.target_properties} -> {target_properties}
            Dim_of_node_vector: {args.dim_of_node_vector}
            Dim_of_edge_vector: {args.dim_of_edge_vector}
            Dim_of_FC         : {args.dim_of_FC}
            Stochastic        : {args.stochastic}
            """)

            shared_model = utils.initialize_model(shared_model, save_fpath)

            scaffolds = [args.scaffold for i in range(args.item_per_cycle)]

            wholes = [None for i in range(args.item_per_cycle)]
            condition1 = target_properties.copy()
            condition2 = scaffold_properties.copy()

            retval_list = [mp.Manager().list() for i in range(args.ncpus)]
            st = time.time()
            processes = []

            for pid in range(args.ncpus):
                p = mp.Process(target=sample,
                               args=(shared_model, wholes, scaffolds, condition1, condition2, pid, retval_list, args))
                p.start()
                processes.append(p)
                time.sleep(0.1)
            for p in processes:
                p.join()
            end = time.time()

            generations = [j[1] for k in retval_list for j in k]
            print(generations)

            mol_obj = Chem.MolFromSmiles(args.scaffold)

            if args.chiral == True:
                smiles = []
                for s in generations:
                    s_ = enumerate_molecule(s)
                    for i in s_:
                        a = Chem.MolFromSmiles(i)
                        if a.GetSubstructMatches(mol_obj, useChirality=True) != ():
                            smiles.append(i)
                with open(args.output_filename, 'w') as output:
                    for smile in smiles:
                        if not '.' in smile:
                            output.write(smile + '\n')
            else:
                with open(args.output_filename, 'w') as output:
                    for smile in generations:
                        if not '.' in smile:
                            output.write(smile + '\n')

    else:
        output_filename = os.path.realpath(os.path.expanduser(args.output_filename))
        target_properties = [normalize(*values) for values in
                             zip(args.target_properties, args.maximum_values, args.minimum_values)]
        scaffold_properties = [normalize(*values) for values in
                               zip(args.scaffold_properties, args.maximum_values, args.minimum_values)]


        mp.set_start_method('spawn')
        torch.manual_seed(1)

        # SSM
        args.N_conditions = len(target_properties) + len(scaffold_properties)

        shared_model = SSM(args)
        shared_model.share_memory()

        print("Model #Params: %dK" % (sum([x.nelement() for x in shared_model.parameters()]) / 1000,))
        print(f"""\
        Ncpus             : {args.ncpus}
        Output_filename   : {args.output_filename }
        OMP_NUM_THREADS   : {os.environ.get('OMP_NUM_THREADS')}
        Num of generations: {args.item_per_cycle} per CPU (Total {args.ncpus * args.item_per_cycle})
        Chiral            : {args.chiral}
        Usrcat            : {args.usrcat}
        Model path        : {save_fpath}
        Output path       : {output_filename}
        Scaffold          : {args.scaffold}
        Scaffold values   : {args.scaffold_properties} -> {scaffold_properties}
        Target values     : {args.target_properties} -> {target_properties}
        Dim_of_node_vector: {args.dim_of_node_vector}
        Dim_of_edge_vector: {args.dim_of_edge_vector}
        Dim_of_FC         : {args.dim_of_FC}
        Stochastic        : {args.stochastic}
        """)

        shared_model = utils.initialize_model(shared_model, save_fpath)


        scaffolds = [args.scaffold for i in range(args.item_per_cycle)]

        wholes = [None for i in range(args.item_per_cycle)]
        condition1 = target_properties.copy()
        condition2 = scaffold_properties.copy()

        retval_list = [mp.Manager().list() for i in range(args.ncpus)]
        st = time.time()
        processes = []

        for pid in range(args.ncpus):
            p = mp.Process(target=sample,
                           args=(shared_model, wholes, scaffolds, condition1, condition2, pid, retval_list, args))
            p.start()
            processes.append(p)
            time.sleep(0.1)
        for p in processes:
            p.join()
        end = time.time()

        generations = [j[1] for k in retval_list for j in k]
        print(generations)

        mol_obj = Chem.MolFromSmiles(args.scaffold)

        if args.chiral == True:
            smiles = []
            for s in generations:
                s_ = enumerate_molecule(s)
                for i in s_:
                    a = Chem.MolFromSmiles(i)
                    if a.GetSubstructMatches(mol_obj, useChirality=True) != ():
                        smiles.append(i)
            with open(args.output_filename, 'w') as output:
                for smile in smiles:
                    if not '.' in smile:
                        output.write(smile + '\n')
        else:
            with open(args.output_filename, 'w') as output:
                for smile in generations:
                    if not '.' in smile:
                        output.write(smile + '\n')
