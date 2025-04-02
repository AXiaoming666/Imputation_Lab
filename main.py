from src import experiment
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset_name', type=str, default='exchange_rate')
    parser.add_argument('--missing_rate', type=float, default=0.5)
    parser.add_argument('--missing_type', type=str, default='MCAR')
    parser.add_argument('--completeness_rate', type=float, default=0.8)
    parser.add_argument('--imputation_method', type=str, default='XGBoost')
    return parser.parse_args()


def main():
    args = parse_arguments()
    config = {
        "seed": args.seed,
        "dataset_name": args.dataset_name,
        "missing_rate": args.missing_rate,
        "missing_type": args.missing_type,
        "completeness_rate": args.completeness_rate,
        "imputation_method": args.imputation_method,
    }
    experiment.main(config)


if __name__ == '__main__':
    main()