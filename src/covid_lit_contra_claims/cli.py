"""Command line interface for covid_lit_contra_claims."""

# -*- coding: utf-8 -*-

import click

from .data.DataLoader import load_train_datasets, load_additional_eval_datasets
from .data.DataExperiments import prepare_training_data
from .models.Training import train_model
from .evaluation.Evaluation import generate_report


@click.command()
@click.option('--output_folder', 'out_dir')
@click.option('--model', 'model')
@click.option('--train_datasets', 'train_datasets')
@click.option('--eval_datasets', 'eval_datasets')
@click.option('--train_prep_experiment', 'train_prep_experiment')
@click.option('--report/--no-report', 'report', default=True)
@click.option('--learning_rate', 'learning_rate', default=1e-6)
@click.option('--batch_size', 'batch_size', default=2)
@click.option('--epochs', 'epochs', default=3)
def main(out_dir, model, train_datasets, eval_datasets, train_prep_experiment, report,
         learning_rate, batch_size, epochs):
    """Run main function."""

    # Load training and evaluation datasets
    train_dataset_dict, val_dataset_dict, test_dataset_dict = load_train_datasets(train_datasets)
    # Two versions of CovidNLI: One where test is a separate network from train
    eval_dataset_dict = load_additional_eval_datasets(eval_datasets)

    # Conduct any input preprocessing for various experiments
    prepared_train_dataset_dict = prepare_training_data(train_dataset_dict, train_prep_experiment)

    # Train model
    training_args = {'epochs': epochs,
                     'batch_size': batch_size,
                     'learning_rate': learning_rate}
    trained_model = train_model(model, prepared_train_dataset_dict, val_dataset_dict,
                                training_args=training_args, out_dir=out_dir)

    if report:
        results_summary = generate_report(trained_model,
                                          test_dataset_dict.update(eval_dataset_dict),
                                          out_dir=out_dir)
        print(f"Summary of training: {results_summary}")


if __name__ == '__main__':
    main()




