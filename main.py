import argparse
from trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--gen_lr", default=0.002, type=float)
    parser.add_argument("--dis_lr", default=0.002, type=float)
    parser.add_argument("--real_label", default=0.9, type=float)
    parser.add_argument("--fake_label", default=0.0, type=float)
    parser.add_argument("--csv_path", default='data/train.csv', type=str)
    args = parser.parse_args()
    trainer = Trainer(
        batch_size=args.batch_size,
        epoch=args.epochs,
        real_label=args.real_label,
        fake_label=args.fake_label,
        gen_lr=args.gen_lr,
        dis_lr=args.dis_lr,
        csv_path=args.csv_path
    )
    trainer.train()


if __name__ == '__main__':
    main()