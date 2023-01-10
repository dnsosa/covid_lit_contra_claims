"""Command line interface for covid_lit_contra_claims."""

# -*- coding: utf-8 -*-

import click

from transformers import AutoTokenizer

from .data.DataExperiments import prepare_training_data, resize_dataset_with_data_ratio
from .data.DataLoader import load_additional_eval_datasets, load_train_datasets
from .data.constants import model_id_mapper
from .evaluation.Evaluation import eval_model_pipeline
from .models.Training import train_model


@click.command()
@click.option('--output_folder', 'out_dir',
              help='Output directory for all analyses.')
@click.option('--model', 'model',
              help='Identifier for pre-trained BERT model (roberta, biobert, bioclinbert, scibert, pubmedbert).')
@click.option('--train_datasets', 'train_datasets',
              help="Identifier indicating which train datasets to use separated by '_' (e.g. multinli_mednli).")
@click.option('--eval_datasets', 'eval_datasets', default="multinli_mednli_mancon_roam_roamPH_roamDD_roamDDPH",
              help="Identifier indicating which eval datasets to use separated by '_' (e.g. roam).")
@click.option('--additional_eval_datasets', 'additional_eval_datasets', default=None,
              help="Identifier indicating which eval datasets to use separated by '_' (e.g. roam).")
@click.option('--truncation/--no-truncation', 'truncation', default=True,
              help="Use truncation in tokenizing and creating HF Datasets?")
@click.option('--train_prep_experiment', 'train_prep_experiment', default="sequential",
              help="Type of experiment for curriculum sequencing ('sequential', 'combined', 'shuffled').")
@click.option('--data_ratios', 'data_ratios', default=None,
              help="Indicates the ratio of training data from one corpus to the next in sequence.")
@click.option('--speed/--no-speed', 'try_speed', default=True,
              help="If true, try to do some HF recommended optimizations for speed.")
@click.option('--report/--no-report', 'report_test', default=False,
              help="If true, generate the report on the test split(s).")
@click.option('--claims_set_id', 'claims_set_id', default=None,
              help="Identifier for set of pairs to make NLI predictions (synth_hcq, claims_subset, all_claims).")
@click.option('--learning_rate', 'learning_rate', default=1e-6,
              help="Learning rate for HF fine-tuning.")
@click.option('--batch_size', 'batch_size', default=2,
              help="Batch size for HF fine-tuning.")
@click.option('--epochs', 'epochs', default=3,
              help="Number of epochs for each dataset in the curriculum during fine-tuning.")
@click.option('--SEED', 'SEED', default=42,
              help="Random seed.")
def main(out_dir, model, train_datasets, eval_datasets, additional_eval_datasets, truncation, train_prep_experiment,
         data_ratios, try_speed, report_test, claims_set_id, learning_rate, batch_size, epochs, SEED):
    """Run main function."""
    # LOAD TOKENIZER
    ################

    if model not in model_id_mapper.keys():
        print(f"Model: '{model}' not a valid transformers model! Must be in: {model_id_mapper.keys()}")
        return None

    # Loading tokenizer here because needed in data loading and model loading
    checkpoint = model_id_mapper[model]
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    # LOAD DATA
    ############

    # Load training datasets
    train_dataset_dict, _, _ = load_train_datasets(train_datasets, tokenizer,
                                                   truncation=truncation,
                                                   SEED=SEED)
    # Load val datasets
    if not report_test:
        # eval = val
        _, eval_dataset_dict, _ = load_train_datasets(eval_datasets, tokenizer,
                                                      truncation=truncation,
                                                      SEED=SEED)
    else:
        # eval = test
        _, _, eval_dataset_dict = load_train_datasets("mancon_manconSS_roam_roamSS", tokenizer,
                                                      truncation=truncation,
                                                      SEED=SEED)

    # Optionally load more test datasets
    if additional_eval_datasets is not None:
        additional_eval_dataset_dict = load_additional_eval_datasets(additional_eval_datasets, tokenizer,
                                                                     truncation=truncation,
                                                                     SEED=SEED)
        eval_dataset_dict.update(additional_eval_dataset_dict)

    # PERTURB DATA
    ################

    # Conduct any input preprocessing for various experiments
    # Note currently only using data_ratio parameter for training data, NOT val data.
    ratio_adjusted_train_dataset_dict = resize_dataset_with_data_ratio(train_dataset_dict,
                                                                       data_ratios=data_ratios,
                                                                       is_train=True,
                                                                       SEED=SEED)
    ratio_adjusted_eval_dataset_dict = resize_dataset_with_data_ratio(eval_dataset_dict,
                                                                      data_ratios=data_ratios,
                                                                      is_train=False,
                                                                      SEED=SEED)
    prepared_train_dataset_dict = prepare_training_data(ratio_adjusted_train_dataset_dict,
                                                        train_prep_experiment,
                                                        SEED)

    # TRAIN MODEL
    #############

    training_args = {'train_datasets': train_datasets,
                     'eval_datasets': eval_datasets,
                     'model': model,
                     'epochs': epochs,
                     'batch_size': batch_size,
                     'learning_rate': learning_rate,
                     'truncation': truncation,
                     'train_prep_experiment': train_prep_experiment,
                     'data_ratios': data_ratios,
                     'SEED': SEED}

    print(f"Running evaluation on {'test' if report_test else 'val'} dataset.")
    trained_model, overall_results = train_model(model,
                                                 tokenizer,
                                                 prepared_train_dataset_dict,
                                                 ratio_adjusted_eval_dataset_dict,
                                                 training_args=training_args,
                                                 try_speed=try_speed,
                                                 out_dir=out_dir,
                                                 SEED=SEED,
                                                 is_test=report_test)

    if claims_set_id is not None:
        eval_model_pipeline(trained_model, tokenizer, out_dir=out_dir, claims_set_id=claims_set_id, SEED=SEED)


if __name__ == '__main__':
    main()
