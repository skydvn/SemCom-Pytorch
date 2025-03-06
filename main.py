import os
import argparse
from train.train_djsccn import DJSCCNTrainer
from train.train_djsccf import DJSCCFTrainer

# from semcom_valid import sem_valid


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds", type=str, default='mnist',
                        help="Dataset")
    parser.add_argument("--recl", type=str, default='mse',
                        help="Reconstruction Loss")
    parser.add_argument("--clsl", type=str, default='ce',
                        help="Classification Loss")
    parser.add_argument("--disl", type=str, default='kl',
                        help="Invariance and Variance Loss")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Inner learning Rate")

    # Loss Setting
    parser.add_argument("--cls-coeff", type=float, default=0.5,
                        help="Coefficient for Classification Loss")
    parser.add_argument("--rec-coeff", type=float, default=1,
                        help="Coefficient for Reconstruction Loss")
    parser.add_argument("--inv-coeff", type=float, default=0.2,
                        help="Coefficient for Invariant Loss")
    parser.add_argument("--var-coeff", type=float, default=0.2,
                        help="Coefficient for Variant Loss")

    # Model Setting
    parser.add_argument("--inv-cdim", type=int, default=32,
                        help="Channel dimension for invariant features")
    parser.add_argument("--var-cdim", type=int, default=32,
                        help="Channel dimension for variant features")

    # VAE Setting
    parser.add_argument("--vae", action="store_true",
                        help="vae switch")
    parser.add_argument("--kld-coeff", type=float, default=0.00025,
                        help="VAE Weight Coefficient")

    # Meta Setting
    parser.add_argument("--bs", type=int, default=1024,
                        help="#batch size")
    parser.add_argument("--tsbs", type=int, default=1024,
                        help="test batch size")
    parser.add_argument("--wk", type=int, default=os.cpu_count(),
                        help="#number of workers")
    parser.add_argument("--out-e", type=int, default=1,
                        help="#number of epochs")
    parser.add_argument("--overall-e", type=int, default=1,
                        help="#number of epochs for overall training")
    parser.add_argument("--irep-e", type=int, default=1,
                        help="#number of epochs for invariant representation learning")
    parser.add_argument("--dv", type=int, default=0,
                        help="Index of GPU")
    parser.add_argument("--model-log", type=bool, default=True,
                        help="save model log")
    parser.add_argument("--operator", type=str, default='window',
                        help="Operator for Pycharm")

    # LOGGING
    parser.add_argument('--wandb', action='store_true',
                        help='toggle to use wandb for online saving')
    parser.add_argument('--log', action='store_true',
                        help='toggle to use tensorboard for offline saving')
    parser.add_argument('--wandb_prj', type=str, default="SemCom-",
                        help='toggle to use wandb for online saving')
    parser.add_argument('--wandb_entity', type=str, default="scalemind",
                        help='toggle to use wandb for online saving')
    parser.add_argument("--verbose", action="store_true",
                        help="printout mode")
    parser.add_argument("--mode", type=str, default="necst",
                        help="necst/djsccf mode")
    args = parser.parse_args()

    if args.mode == "djsccf":
        trainer = DJSCCNTrainer(args=args)
    elif args.mode == "djsccn":
        trainer = DJSCCFTrainer(args=args)

    trainer.train()
    trainer.evaluate_semantic_communication()
