"""Command line interface for covid_lit_contra_claims."""

# -*- coding: utf-8 -*-

import click

#from .data.DataLoader import __
#from .data.Preprocessing import __


@click.command()
@click.option('--output_folder', 'out_dir')
@click.option('--model', 'model')
@click.option('--dataset', 'dataset')
@click.option('--sequenced/--no-sequenced', 'sequenced')
@click.option('--bonus_train', 'bonus_train')
@click.option('--report/--no-report', 'report', default=True)
@click.option('--learning_rate', 'learning_rate', default=1e-6)
@click.option('--batch_size', 'batch_size', default=2)
@click.option('--epochs', 'epochs', default=3)
def main(out_dir, model, dataset, sequenced, bonus_train, report, learning_rate, batch_size, epochs):
    """Run main function."""


if __name__ == '__main__':
    main()




