"""Command line interface for covid_lit_contra_claims."""

# -*- coding: utf-8 -*-

import click

from transformers import AutoTokenizer

from .data.constants import model_id_mapper
from .data.DataLoader import load_train_datasets, load_additional_eval_datasets
from .data.DataExperiments import prepare_training_data
from .models.Training import train_model
from .evaluation.Evaluation import generate_report


@click.command()
@click.option('--output_folder', 'out_dir')
@click.option('--model', 'model')
@click.option('--train_datasets', 'train_datasets')
@click.option('--eval_datasets', 'eval_datasets')
@click.option('--truncation/--no-truncation', 'truncation', default=True)
@click.option('--train_prep_experiment', 'train_prep_experiment', default="sequential")
@click.option('--data_ratios', 'data_ratios', default=None)
@click.option('--report/--no-report', 'report', default=False)
@click.option('--learning_rate', 'learning_rate', default=1e-6)
@click.option('--batch_size', 'batch_size', default=2)
@click.option('--epochs', 'epochs', default=3)
@click.option('--SEED', 'SEED', default=42)
def main(out_dir, model, train_datasets, eval_datasets, truncation, train_prep_experiment, data_ratios, report,
         learning_rate, batch_size, epochs, SEED):
    """Run main function."""

    if model not in model_id_mapper.keys():
        print(f"Model: '{model}' not a valid transformers model! Must be in: {model_id_mapper.keys()}")
        return None

    # Loading tokenizer here because needed in data loading and model loading
    checkpoint = model_id_mapper[model]
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    # Load training and evaluation datasets
    train_dataset_dict, val_dataset_dict, test_dataset_dict = load_train_datasets(train_datasets, tokenizer,
                                                                                  truncation=truncation,
                                                                                  SEED=SEED)

    eval_dataset_dict = load_additional_eval_datasets(eval_datasets, tokenizer,
                                                      truncation=truncation,
                                                      SEED=SEED)

    # Conduct any input preprocessing for various experiments
    # Note currently only using data_ratio parameter for training data, NOT val data.
    prepared_train_dataset_dict = prepare_training_data(train_dataset_dict, train_prep_experiment, data_ratios, SEED)

    # Train model
    training_args = {'train_datasets': train_datasets,
                     'eval_datasets': eval_datasets,
                     'epochs': epochs,
                     'batch_size': batch_size,
                     'learning_rate': learning_rate,
                     'truncation': truncation,
                     'train_prep_experiment': train_prep_experiment,
                     'data_ratios': data_ratios}
    trained_model, overall_results = train_model(model, tokenizer, prepared_train_dataset_dict, val_dataset_dict,
                                                 training_args=training_args, out_dir=out_dir, SEED=SEED)

    # Final report--test set statistics
    if report:
        results_summary = generate_report(trained_model,
                                          test_dataset_dict.update(eval_dataset_dict),
                                          out_dir=out_dir)
        print(f"Summary of training: {results_summary}")


if __name__ == '__main__':
    main()




