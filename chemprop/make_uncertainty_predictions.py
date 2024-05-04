import os
import re
import logging
import time
from typing import List, Union, Tuple, Literal

import numpy as np
import torch
import pandas as pd

import chemprop
from chemprop.args import PredictArgs, TrainArgs
from chemprop.data import StandardScaler, AtomBondScaler
from chemprop.models import MoleculeModel
from chemprop.train.make_predictions import load_model, set_features, load_data
from chemprop.train.predict import predict


def make_uncertainty_predictions(args: PredictArgs,
                                 smiles: List[List[str]] = None,
                                 model_objects: Tuple[PredictArgs,
                                                      TrainArgs,
                                                      List[MoleculeModel],
                                                      List[Union[StandardScaler, AtomBondScaler]],
                                                      int,
                                                      List[str]] = None,
                                 ensemble_size: int = None,
                                 return_raw_pred: bool = False):
    """
    Loads data and a trained model and uses the model to make predictions on the data.

    If SMILES are provided, then makes predictions on smiles.
    Otherwise makes predictions on :code:`args.test_data`.

    :param args: A :class:`~chemprop.args.PredictArgs` object containing arguments for
                loading data and a model and making predictions.
    :param smiles: List of list of SMILES to make predictions on.
    :param model_objects: Tuple of output of load_model function which can be called separately outside this function. Preloaded model objects should have
                used the non-generator option for load_model if the objects are to be used multiple times or are intended to be used for calibration as well.
    :param ensemble_size: The number of ensemble models. if not specified, load from checkpoint files
    :param return_raw_pred: return the raw predictions if not, return the calculated data/model uncertainty
    :return: A list of lists of target predictions. If returning uncertainty, a tuple containing first prediction values then uncertainty estimates.
    """
    if model_objects:
        (
            args,
            train_args,
            models,
            scalers,
            num_tasks,
            task_names,
        ) = model_objects
    else:
        (
            args,
            train_args,
            models,
            scalers,
            num_tasks,
            task_names,
        ) = load_model(args, generator=True)

    set_features(args, train_args)

    # Note: to get the invalid SMILES for your data, use the get_invalid_smiles_from_file or
    # get_invalid_smiles_from_list functions from data/utils.py
    full_data, test_data, test_data_loader, full_to_valid_indices = load_data(
        args, smiles
    )
    if ensemble_size is not None:
        ens = ensemble_size
    else:
        ens = int(args.ensemble_size / args.num_folds)
    predict_results = [list([None] * ens) for _ in range(args.num_folds)]
    # return uncertainty if loss function in mve and evidential
    if args.loss_function in ['mve', "evidential"]:
        return_unc = True
    else:
        return_unc = False

    for model, scaler_list, path in zip(models, scalers, args.checkpoint_paths):
        (
            scaler,
            features_scaler,
            atom_descriptor_scaler,
            bond_descriptor_scaler,
            atom_bond_scaler,
        ) = scaler_list
        if (
                features_scaler is not None
                or atom_descriptor_scaler is not None
                or bond_descriptor_scaler is not None
        ):
            test_data.reset_features_and_targets()
            if features_scaler is not None:
                test_data.normalize_features(features_scaler)
            if atom_descriptor_scaler is not None:
                test_data.normalize_features(
                    atom_descriptor_scaler, scale_atom_descriptors=True
                )
            if bond_descriptor_scaler is not None:
                test_data.normalize_features(
                    bond_descriptor_scaler, scale_bond_descriptors=True
                )

        predictions = predict(model=model,
                              data_loader=test_data_loader,
                              scaler=scaler,
                              atom_bond_scaler=atom_bond_scaler,
                              return_unc_parameters=return_unc,
                              )
        fold_ith, model_ith = None, None

        # add predictions according to the order of fold and model
        for name in reversed(path.split(os.sep)):
            if 'model' in name and '.pt' in name:
                continue
            elif 'model' in name and '.pt' not in name:
                model_ith = re.findall(r'\d+', name)[0]

            if 'fold' in name:
                fold_ith = re.findall(r'\d+', name)[0]

            if model_ith and fold_ith:
                predict_results[int(fold_ith)][int(model_ith)] = predictions
                break

    if return_raw_pred:
        return predict_results

    unc_results = []
    for fold_results in predict_results:
        fold_all_unc_results = []
        for results in fold_results:
            if results is not None:
                if args.loss_function == 'evidential':
                    preds, lambdas, alphas, betas = results
                    model_var = np.array(betas) / (np.array(lambdas) * (np.array(alphas) - 1))
                    data_var = np.array(betas) / (np.array(alphas) - 1)
                    fold_unc_results = np.concatenate([preds, data_var, model_var], axis=-1)
                elif args.loss_function == 'mve':
                    fold_unc_results = np.concatenate(results, axis=-1)
                else:
                    fold_unc_results = results
                fold_all_unc_results.append(fold_unc_results)
        # averaging results if model has
        if args.loss_function == 'evidential':
            unc_results.append(np.average(np.array(fold_all_unc_results), axis=0).tolist())
        if args.loss_function in ['mve', 'mse']:
            model_var = np.var(np.array(fold_all_unc_results), axis=0)[:, 0, np.newaxis]
            ave = np.average(np.array(fold_all_unc_results), axis=0)
            unc_results.append(np.concatenate([ave, model_var], axis=1).tolist())
        else:
            raise KeyError(f'Unknown loss function {args.loss_function}. Using standard code can avoid this error')

    return unc_results


class Estimator:
    """
    Estimating reaction kinetics
    """

    def __init__(self,
                 checkpoint_path: str,
                 seed: int = 0,
                 feature_gen: str = None,
                 uncertainty_method: Literal['ensemble', 'dropout'] = None,
                 dropout_sampling_size: int = None,
                 uncertainty_dropout_p: float = None,
                 gpu: int = None,
                 logger: logging.Logger = None,
                 ensemble_size: int = None,
                 return_raw_pred: bool = False):
        """
        Initialize the estimator
        Args:
            seed:
            checkpoint_path: The machine learning model path
            uncertainty_method: dropout, ensemble, evidential, mve
        """
        self.uncertainty_method = uncertainty_method
        self.check_point_path = checkpoint_path

        self.feature_generator = feature_gen

        if uncertainty_method == 'dropout':
            self.dropout_sampling_size = dropout_sampling_size
            self.uncertainty_dropout_p = uncertainty_dropout_p
        else:
            self.dropout_sampling_size = None

        self.logger = logger

        self.gpu = gpu
        if self.uncertainty_method == 'dropout':
            torch.manual_seed(seed)

        self.ensemble_size = ensemble_size
        self.return_raw_pred = return_raw_pred

    def estimate(self, smiles: List[str], features: List = None):
        """
        Estimate reaction predictions
        Args:
            smiles: reaction smiles
            features:
        Returns:
        """
        df_sm = pd.DataFrame(data=smiles, columns=['rxn'])
        df_sm.to_csv('_rxns.csv', index=False)
        exist_smis_path = False
        while not exist_smis_path:
            if os.path.exists('_rxns.csv'):
                exist_smis_path = True
            else:
                time.sleep(0.1)
        if features is not None:
            df_feat = pd.DataFrame(data=features, columns=['feat'])
            df_feat.to_csv('_feats.csv', index=False)
            exist_feat_path = False
            while not exist_feat_path:
                if os.path.exists('_feats.csv'):
                    exist_feat_path = True
                else:
                    time.sleep(0.1)

        arguments = [
            '--test_path', '_rxns.csv',
            '--preds_path', '/dev/null',
            '--checkpoint_dir', self.check_point_path,
        ]
        if features:
            arguments.extend(['--features_path', '_feats.csv'])
        if self.feature_generator is not None:
            if self.feature_generator == 'rdkit_2d_normalized':
                arguments.extend(['--features_generator', 'rdkit_2d_normalized', '--no_features_scaling'])
            else:
                arguments.extend(['--features_generator', self.feature_generator])
        if self.uncertainty_method == 'dropout' and self.dropout_sampling_size is not None:
            arguments.extend(['--dropout_sampling_size', str(self.dropout_sampling_size)])
            arguments.extend(['--uncertainty_dropout_p', str(self.uncertainty_dropout_p)])

        args = chemprop.args.PredictArgs().parse_args(arguments)
        preds = make_uncertainty_predictions(args=args,
                                             ensemble_size=self.ensemble_size,
                                             return_raw_pred=self.return_raw_pred)

        os.remove('_rxns.csv')
        if features is not None:
            os.remove('_feats.csv')

        return preds