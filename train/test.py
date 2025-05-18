class DGSCTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)

        self.model = DGSC_CIFAR(self.args, self.in_channel, self.class_num).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=self.args.lr)
        self.criterion = DGSCLoss()
        self.domain_list = ['AWGN']

    def train(self):
        domain_list = self.domain_list
        rec = [ [] for _ in domain_list ]
        for epoch in range(self.args.out_e):
            epoch_train_loss = 0
            epoch_val_loss = 0
            channel_loss_sums = [0] * len(domain_list)  # To accumulate losses for each channel
            channel_loss_counts = [0] * len(domain_list)  # To count occurrences for each channel
            
            self.model.train()
            for x, y in tqdm(self.train_dl):
                x, y = x.to(self.device), y.to(self.device)
                channel_losses = []  # Array to store losses for each channel
                total_loss = 0

                for i, domain_str in enumerate(domain_list):
                    out = self.model.channel_perturb(x, domain_str)
                    loss = self.criterion.forward(self.args, out, x)
                    channel_losses.append(loss.item())  # Store individual channel loss
                    channel_loss_sums[i] += loss.item()  # Accumulate loss for this channel
                    channel_loss_counts[i] += 1  # Increment count for this channel
                    total_loss += loss  # Accumulate total loss
                    rec[i].append(out)

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
            
                epoch_train_loss += total_loss.detach().item()
                print(f'Channel Losses: {channel_losses}')  # Print losses for each batch

            epoch_train_loss /= (len(self.train_dl))
            print('Epoch Loss:', epoch_train_loss)
            self.writer.add_scalar('train/_loss', epoch_train_loss, epoch)

            # Calculate and print average loss for each channel
            avg_channel_losses = [channel_loss_sums[i] / channel_loss_counts[i] for i in range(len(domain_list))]
            print(f'Average Channel Losses: {avg_channel_losses}')

            self.model.eval()
            with torch.no_grad():
                for test_imgs, test_labels in tqdm(self.test_dl):
                    test_imgs, test_labels = test_imgs.to(self.device), test_labels.to(self.device)
                    test_rec = self.model(test_imgs)
                    loss = self.criterion.forward(self.args, test_imgs, test_rec)
                    epoch_val_loss += loss.detach().item()
                epoch_val_loss /= (len(self.test_dl))
                self.writer.add_scalar('val/_loss', epoch_val_loss, epoch)

            # Saving checkpoint
            self.save_model(epoch=epoch, model=self.model)

        self.writer.close()
        self.save_config()