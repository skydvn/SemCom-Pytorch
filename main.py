import os
import argparse
from train.train_djsccn import DJSCCNTrainer
from train.train_djsccf import DJSCCFTrainer

# from semcom_valid import sem_valid


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default='./out',
                         help="Path to save outputs")
    parser.add_argument("--ds", type=str, default='mnist',
                        help="Dataset")
    parser.add_argument("--base_snr", type=float, default=10,
                        help="SNR during train")
    # parser.add_argument('--snr_list', default=['19', '13',
    #                     '7', '4', '1'], nargs='+', help='snr_list')
    parser.add_argument('--ratio_list', default=['1/6', '1/12'], nargs='+', help='ratio_list')
    parser.add_argument('--channel_type', default='AWGN', type=str,
                        choices=['AWGN', 'Rayleigh'], help='channel')
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
    parser.add_argument("--bs", type=int, default=128,
                        help="#batch size")
    parser.add_argument("--wk", type=int, default=os.cpu_count(),
                        help="#number of workers")
    parser.add_argument("--out-e", type=int, default=1,
                        help="#number of epochs")
    parser.add_argument("--dv", type=int, default=0,
                        help="Index of GPU")
    parser.add_argument("--device", type=bool, default=True,
                        help="Return device or not")
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
    parser.add_argument("--algo", type=str, default="djsccn",
                        help="necst/djsccf mode")
    args = parser.parse_args()

    if args.algo == "djsccf":
        trainer = DJSCCFTrainer(args=args)
    elif args.algo == "djsccn":
        trainer = DJSCCNTrainer(args=args)
    else:
        raise ValueError("Invalid trainer")

    trainer.train()

    config_dir = os.path.join(args.out, 'configs')
    config_files = [os.path.join(config_dir, name) for name in os.listdir(config_dir)
                    if (args.ds in name or args.ds.upper() in name) and args.channel_type in name and name.endswith('.yaml')]
    output_dir = args.out
    """
        Switch cases
        * If train_flag == False:
            - If full_flag == False: 
                run the input path 
            - Elif full_flag == True:
                for config_path in config_files:
                    trainer.evaluate(
                        config_path=config_path,
                        output_dir=output_dir
                    )
        * If train_flag == True:
                run the newly saved path
    """

    for config_path in config_files:
        trainer.evaluate(
            config_path=config_path,
            output_dir=output_dir
        )

