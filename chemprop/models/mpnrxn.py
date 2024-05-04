from typing import List, Union, Tuple
from functools import reduce

import numpy as np
from rdkit import Chem
import torch
import torch.nn as nn

from chemprop.args import TrainArgs
from chemprop.features import BatchReactionGraph, get_atom_fdim, get_bond_fdim, mol2graph
from chemprop.nn_utils import index_select_ND, get_activation_function
from chemprop.models.attention import AtomAttention


class MPNEncoder(nn.Module):
    """An :class:`MPNEncoder` is a message passing neural network for encoding a molecule."""

    def __init__(self, args: TrainArgs, atom_fdim: int, bond_fdim: int, hidden_size: int = None,
                 bias: bool = None, depth: int = None):
        """
        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        :param atom_fdim: Atom feature vector dimension.
        :param bond_fdim: Bond feature vector dimension.
        :param hidden_size: Hidden layers dimension.
        :param bias: Whether to add bias to linear layers.
        :param depth: Number of message passing steps.
       """
        super(MPNEncoder, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.atom_messages = args.atom_messages
        self.hidden_size = hidden_size or args.hidden_size
        self.bias = bias or args.bias
        self.depth = depth or args.depth
        self.layers_per_message = 1
        self.undirected = args.undirected
        self.device = args.device
        self.aggregation = args.aggregation
        self.aggregation_norm = args.aggregation_norm
        self.is_atom_bond_targets = args.is_atom_bond_targets
        self.dual_react_prod_model = args.dual_react_prod_model

        if self.atom_messages:
            raise KeyError('Atom message are currently not supported for cross attention')

        # Cached zeros
        self.cached_zero_vector = nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False)

        # Dropout
        self.dropout = nn.Dropout(args.dropout)

        # Activation
        self.act_func = get_activation_function(args.activation)

        # Input dim
        input_dim = self.atom_fdim if self.atom_messages else self.bond_fdim

        if self.atom_messages:
            w_h_input_size = self.hidden_size + self.bond_fdim
        else:
            w_h_input_size = self.hidden_size

        # initialize MPN input and input attention layers
        self.W_i = nn.Linear(input_dim, self.hidden_size, bias=self.bias)

        # initialize MPN hidden and hidden attention layers
        self.h_cross_att = AtomAttention(input_size=self.hidden_size, output_size=self.hidden_size,
                                         cached_vector=self.cached_zero_vector, mol_feature=False,
                                         bias=self.bias, activation_func=self.act_func, dropout_func=self.dropout)
        self.W_h = nn.Linear(w_h_input_size, self.hidden_size, bias=self.bias)

        self.W_h_e = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_o = nn.Linear(self.atom_fdim + self.hidden_size, self.hidden_size)

        if args.atom_descriptors == 'descriptor':
            self.atom_descriptors_size = args.atom_descriptors_size
            self.atom_descriptors_layer = nn.Linear(self.hidden_size + self.atom_descriptors_size,
                                                    self.hidden_size + self.atom_descriptors_size)

        if args.bond_descriptors == 'descriptor':
            self.bond_descriptors_size = args.bond_descriptors_size
            self.bond_descriptors_layer = nn.Linear(self.hidden_size + self.bond_descriptors_size,
                                                    self.hidden_size + self.bond_descriptors_size)

    def get_rxn_encoding(self, react_message: torch.Tensor, prod_message: torch.Tensor) -> torch.Tensor:
        """
        Get reaction encoding

        :param react_message: reactant message
        :param prod_message: product message
        :return: reaction encoding
        """
        diff_hidden = react_message - prod_message
        if self.dual_react_prod_model == 'reac_prod':
            features = react_message + prod_message
        elif self.dual_react_prod_model == 'reac_diff':
            features = react_message + diff_hidden
        elif self.dual_react_prod_model == 'prod_diff':
            features = prod_message + diff_hidden
        else:
            features = diff_hidden

        return features

    def dual_message_passing(self,
                             rf_atoms: torch.Tensor,
                             pf_atoms: torch.Tensor,
                             rf_bonds: torch.Tensor,
                             pf_bonds: torch.Tensor,
                             b2revb: torch.Tensor,
                             a2b: torch.Tensor,
                             b2a: torch.Tensor,
                             b_scope: List,
                             ):
        rinput = self.W_i(rf_bonds)  # num_bonds x hidden_size
        pinput = self.W_i(pf_bonds)
        rmessage = self.act_func(rinput)  # num_bonds x hidden_size
        pmessage = self.act_func(pinput)

        # Message passing
        for depth in range(self.depth - 1):
            # m(a1 -> a2) = [sum_{a0 \in nei(a1)} m(a0 -> a1)] - m(a2 -> a1)
            # message      a_message = sum(nei_a_message)      rev_message
            nei_ra_message = index_select_ND(rmessage, a2b)  # num_atoms x max_num_bonds x hidden
            ra_message = nei_ra_message.sum(dim=1)  # num_atoms x hidden
            rev_rmessage = rmessage[b2revb]  # num_bonds x hidden
            rmessage = ra_message[b2a] - rev_rmessage  # num_bonds x hidden

            nei_pa_message = index_select_ND(pmessage, a2b)  # num_atoms x max_num_bonds x hidden
            pa_message = nei_pa_message.sum(dim=1)  # num_atoms x hidden
            rev_pmessage = pmessage[b2revb]  # num_bonds x hidden
            pmessage = pa_message[b2a] - rev_pmessage  # num_bonds x hidden

            rmessage = self.W_h(rmessage)
            rmessage = self.act_func(rinput + rmessage)  # num_bonds x hidden_size
            rmessage = self.dropout(rmessage)  # num_bonds x hidden

            pmessage = self.W_h(pmessage)
            pmessage = self.act_func(pinput + pmessage)  # num_bonds x hidden_size
            pmessage = self.dropout(pmessage)  # num_bonds x hidden

            r_cross_message = self.h_cross_att(pmessage, b_scope, rmessage) + rmessage
            p_cross_message = self.h_cross_att(rmessage, b_scope, pmessage) + pmessage
            rmessage, pmessage = r_cross_message, p_cross_message

        # get reaction messages and redo message passing process
        rxn_message = self.get_rxn_encoding(rmessage, pmessage)
        rxn_input = self.get_rxn_encoding(rinput, pinput)
        rxn_atom_features = self.get_rxn_encoding(rf_atoms, pf_atoms)
        for depth in range(self.depth - 1):
            nei_a_message = index_select_ND(rxn_message, a2b)  # num_atoms x max_num_bonds x hidden
            a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
            rev_message = rxn_message[b2revb]  # num_bonds x hidden
            rxn_message = a_message[b2a] - rev_message  # num_bonds x hidden

            rxn_message = self.W_h_e(rxn_message)
            rxn_message = self.act_func(rxn_input + rxn_message)  # num_bonds x hidden_size
            rxn_message = self.dropout(rxn_message)  # num_bonds x hidden

        nei_a_message = index_select_ND(rxn_message, a2b)  # num_atoms x max_num_bonds x hidden
        a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
        a_input = torch.cat([rxn_atom_features, a_message], dim=1)  # num_atoms x (atom_fdim + hidden)

        rxn_atom_hiddens = self.act_func(self.W_o(a_input))  # num_atoms x hidden
        rxn_atom_hiddens = self.dropout(rxn_atom_hiddens)  # num_atoms x hidden

        return rxn_atom_hiddens

    def forward(self,
                mol_graph: BatchReactionGraph) -> Union[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Encodes a batch of molecular graphs.

        :param mol_graph: A :class:`~chemprop.features.featurization.BatchMolGraph` representing
                          a batch of molecular graphs.
        :return: A PyTorch tensor of shape :code:`(num_molecules, hidden_size)` containing the encoding of each molecule.
        """
        rf_atoms, pf_atoms, rf_bonds, pf_bonds, a2b, b2a, b2revb, a_scope, b_scope \
            = mol_graph.get_components(atom_messages=self.atom_messages)
        # to device
        rf_atoms, rf_bonds, pf_atoms, pf_bonds, a2b, b2a, b2revb = rf_atoms.to(self.device), rf_bonds.to(self.device), \
            pf_atoms.to(self.device), pf_bonds.to(self.device), a2b.to(self.device), b2a.to(self.device), \
            b2revb.to(self.device)

        features = self.dual_message_passing(rf_atoms=rf_atoms,
                                             pf_atoms=pf_atoms,
                                             rf_bonds=rf_bonds,
                                             pf_bonds=pf_bonds,
                                             b2revb=b2revb,
                                             a2b=a2b,
                                             b2a=b2a,
                                             b_scope=b_scope)

        # Readout
        mol_vecs = []
        for i, (ra_start, ra_size) in enumerate(a_scope):
            if ra_size == 0:
                mol_vecs.append(self.cached_zero_vector)
            else:
                cur_hiddens = features.narrow(0, ra_start, ra_size)
                mol_vec = cur_hiddens  # (num_atoms, hidden_size)
                if self.aggregation == 'mean':
                    mol_vec = mol_vec.sum(dim=0) / ra_size
                elif self.aggregation == 'sum':
                    mol_vec = mol_vec.sum(dim=0)
                elif self.aggregation == 'norm':
                    mol_vec = mol_vec.sum(dim=0) / self.aggregation_norm
                mol_vecs.append(mol_vec)

        mol_vecs = torch.stack(mol_vecs, dim=0)  # (num_molecules, hidden_size)

        return mol_vecs  # num_molecules x hidden


class RXNMPN(nn.Module):
    """An :class:`MPN` is a wrapper around :class:`MPNEncoder` which featurizes input as needed."""

    def __init__(self,
                 args: TrainArgs,
                 atom_fdim: int = None,
                 bond_fdim: int = None):
        """
        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        :param atom_fdim: Atom feature vector dimension.
        :param bond_fdim: Bond feature vector dimension.
        """
        super(RXNMPN, self).__init__()
        is_reaction = args.reaction if not args.react_prod_cross_attention else False
        self.reaction = args.reaction
        self.reaction_solvent = args.reaction_solvent
        self.atom_fdim = atom_fdim or get_atom_fdim(overwrite_default_atom=args.overwrite_default_atom_features,
                                                    is_reaction=is_reaction if is_reaction is not False else self.reaction_solvent)
        self.bond_fdim = bond_fdim or get_bond_fdim(overwrite_default_atom=args.overwrite_default_atom_features,
                                                    overwrite_default_bond=args.overwrite_default_bond_features,
                                                    atom_messages=args.atom_messages,
                                                    is_reaction=is_reaction if is_reaction is not False else self.reaction_solvent)
        self.features_only = args.features_only
        self.use_input_features = args.use_input_features
        self.device = args.device
        self.atom_descriptors = args.atom_descriptors
        self.bond_descriptors = args.bond_descriptors
        self.overwrite_default_atom_features = args.overwrite_default_atom_features
        self.overwrite_default_bond_features = args.overwrite_default_bond_features
        self.individual_react_prod_mol = args.react_prod_cross_attention

        if self.features_only:
            return

        if not self.reaction_solvent:
            if args.mpn_shared:
                self.encoder = nn.ModuleList(
                    [MPNEncoder(args, self.atom_fdim, self.bond_fdim)] * args.number_of_molecules)
            else:
                self.encoder = nn.ModuleList([MPNEncoder(args, self.atom_fdim, self.bond_fdim)
                                              for _ in range(args.number_of_molecules)])
        else:
            self.encoder = MPNEncoder(args, self.atom_fdim, self.bond_fdim)
            # Set separate atom_fdim and bond_fdim for solvent molecules
            self.atom_fdim_solvent = get_atom_fdim(overwrite_default_atom=args.overwrite_default_atom_features,
                                                   is_reaction=False)
            self.bond_fdim_solvent = get_bond_fdim(overwrite_default_atom=args.overwrite_default_atom_features,
                                                   overwrite_default_bond=args.overwrite_default_bond_features,
                                                   atom_messages=args.atom_messages,
                                                   is_reaction=False)
            self.encoder_solvent = MPNEncoder(args, self.atom_fdim_solvent, self.bond_fdim_solvent,
                                              args.hidden_size_solvent, args.bias_solvent, args.depth_solvent)

    def forward(self,
                batch: Union[List[List[str]], List[List[Chem.Mol]], List[List[Tuple[Chem.Mol, Chem.Mol]]], List[
                    BatchReactionGraph]],
                features_batch: List[np.ndarray] = None,
                atom_descriptors_batch: List[np.ndarray] = None,
                atom_features_batch: List[np.ndarray] = None,
                bond_descriptors_batch: List[np.ndarray] = None,
                bond_features_batch: List[np.ndarray] = None) -> torch.Tensor:
        """
        Encodes a batch of molecules.

        :param batch: A list of list of SMILES, a list of list of RDKit molecules, or a
                      list of :class:`~chemprop.features.featurization.BatchMolGraph`.
                      The outer list or BatchMolGraph is of length :code:`num_molecules` (number of datapoints in batch),
                      the inner list is of length :code:`number_of_molecules` (number of molecules per datapoint).
        :param features_batch: A list of numpy arrays containing additional features.
        :param atom_descriptors_batch: A list of numpy arrays containing additional atom descriptors.
        :param atom_features_batch: A list of numpy arrays containing additional atom features.
        :param bond_descriptors_batch: A list of numpy arrays containing additional bond descriptors.
        :param bond_features_batch: A list of numpy arrays containing additional bond features.
        :return: A PyTorch tensor of shape :code:`(num_molecules, hidden_size)` containing the encoding of each molecule.
        """
        if type(batch[0]) != BatchReactionGraph:
            # Group first molecules, second molecules, etc for mol2graph
            batch = [[mols[i] for mols in batch] for i in range(len(batch[0]))]

            # TODO: handle atom_descriptors_batch with multiple molecules per input
            if self.atom_descriptors == 'feature':
                if len(batch) > 1:
                    raise NotImplementedError('Atom/bond descriptors are currently only supported with one molecule '
                                              'per input (i.e., number_of_molecules = 1).')

                batch = [
                    mol2graph(
                        mols=b,
                        atom_features_batch=atom_features_batch,
                        bond_features_batch=bond_features_batch,
                        overwrite_default_atom_features=self.overwrite_default_atom_features,
                        overwrite_default_bond_features=self.overwrite_default_bond_features,
                    )
                    for b in batch
                ]
            elif self.bond_descriptors == 'feature':
                if len(batch) > 1:
                    raise NotImplementedError('Atom/bond descriptors are currently only supported with one molecule '
                                              'per input (i.e., number_of_molecules = 1).')

                batch = [
                    mol2graph(
                        mols=b,
                        bond_features_batch=bond_features_batch,
                        overwrite_default_atom_features=self.overwrite_default_atom_features,
                        overwrite_default_bond_features=self.overwrite_default_bond_features
                    )
                    for b in batch
                ]
            else:
                batch = [mol2graph(b, individual_react_prod_mol=self.individual_react_prod_mol) for b in batch]

        if self.use_input_features:
            features_batch = torch.from_numpy(np.stack(features_batch)).float().to(self.device)

            if self.features_only:
                return features_batch

        if self.atom_descriptors == 'descriptor' or self.bond_descriptors == 'descriptor':
            if len(batch) > 1:
                raise NotImplementedError('Atom descriptors are currently only supported with one molecule '
                                          'per input (i.e., number_of_molecules = 1).')

            encodings = [enc(ba, atom_descriptors_batch, bond_descriptors_batch) for enc, ba in
                         zip(self.encoder, batch)]
        else:
            if not self.reaction_solvent:
                encodings = [enc(ba) for enc, ba in zip(self.encoder, batch)]
            else:
                encodings = []
                for ba in batch:
                    if ba.is_reaction:
                        encodings.append(self.encoder(ba))
                    else:
                        encodings.append(self.encoder_solvent(ba))

        output = encodings[0] if len(encodings) == 1 else torch.cat(encodings, dim=1)

        if self.use_input_features:
            if len(features_batch.shape) == 1:
                features_batch = features_batch.view(1, -1)

            output = torch.cat([output, features_batch], dim=1)

        return output
