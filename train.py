def train_necst(args: argparse):
    # Folder Setup
    runs_dir = os.getcwd() + "/runs"
    if not os.path.exists(runs_dir):
        os.mkdir(runs_dir)
    ds_dir = runs_dir + f"/{args.ds}"
    if not os.path.exists(ds_dir):
        os.mkdir(ds_dir)
    bs_dir = ds_dir + f"/{args.bs}_{args.out_e}"
    if not os.path.exists(bs_dir):
        os.mkdir(bs_dir)
    bs_dir_len = len(next(os.walk(bs_dir))[1])
    if bs_dir_len == 0:
        exp_dir = bs_dir + "/exp0"
    else:
        old_exp_dir = bs_dir + f"/exp{bs_dir_len - 1}"
        if os.path.exists(old_exp_dir) and len(glob(old_exp_dir + "/*.parquet")) < 2:
            shutil.rmtree(path=old_exp_dir)
            exp_dir = old_exp_dir
        else:
            exp_dir = bs_dir + f"/exp{bs_dir_len}"
    os.mkdir(exp_dir)

    # Path settings
    best_model_path = exp_dir + f"/best.pt"
    last_model_path = exp_dir + f"/last.pt"
    log_train_path = exp_dir + "/train_log.parquet"
    log_test_path = exp_dir + "/test_log.parquet"
    config_path = exp_dir + "/config.json"

    coeff_dict = {
        "epoch": 1,
        "cls_loss": args.cls_coeff,
        "rec_loss": args.rec_coeff,
        "psnr_loss": 1,
        "kld_loss": args.kld_coeff,
        "inv_loss": args.inv_coeff,
        "var_loss": args.var_coeff,
        "irep_loss": 1,
        "total_loss": 1,
        "accuracy": 1
    }

    # Setup
    log_interface = Logging(args)

    (train_dl, test_dl, valid_dl), args = get_ds(args)

    print(len(train_dl), len(test_dl))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=0)
    # device = torch.device("cpu", index=0)

    # model = CNN_EMnist_VAE().to(device=device)
    if args.ds == "emnist":
        in_channel = 1
        class_num = 10
    elif args.ds == "cifar10":
        in_channel = 3
        class_num = 10
    else:
        in_channel = 3
        class_num = 10

    # model = CNN_EMnist_NoVAE(in_channel, class_num).to(device=device)
    print(args.vae)
    # model = model_mapping[args.ds](in_channel, class_num).to(device=device)
    model = model_mapping[args.ds](args, in_channel, class_num).to(device=device)
    print(model)
    ukie_optimizer = Adam(model.parameters(), lr=args.lr)
    inv_optimizer = Adam([
        {'params': model.inv.parameters()},
        {'params': model.classifier.parameters()}], lr=args.lr)

    criterion = InvLoss(
        rec_loss=args.recl,
        cls_loss=args.clsl,
        dis_loss=args.disl
    )

    # Training
    old_loss_value = 1e26
    old_acc_value = 0


def train_djsccf(args):
    pass
