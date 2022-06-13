import argparse
from trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--csv_path", default='data/train.csv', type=str)
    args = parser.parse_args()
    trainer = Trainer(
        batch_size=args.batch_size,
        epoch=args.epochs,
        csv_path=args.csv_path
    )
    trainer.train()


if __name__ == '__main__':
    main()