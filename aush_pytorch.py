# need to deal with some variable

class AUSH(Attacker):

    def __init__(self):
        super(AUSH, self).__init__()
        # self.selected_ids = list(map(int, self.args.selected_ids.split(',')))
        # self.selected_ids = [1551, 2510, 1167, 2362, 2233, 2801, 905, 1976, 3239, 3262]
        self.selected_ids = [3530]
        #
        self.restore_model = self.args.restore_model
        self.model_path = self.args.model_path
        #
        # 
        # self.epochs = self.args.epoch
        self.epochs = 100
        self.batch_size = self.args.batch_size
        #
        self.learning_rate_G = self.args.learning_rate_G
        self.reg_rate_G = self.args.reg_rate_G
        self.ZR_ratio = self.args.ZR_ratio
        #
        self.learning_rate_D = self.args.learning_rate_D
        self.reg_rate_D = self.args.reg_rate_D
        #
        self.verbose = self.args.verbose
        self.T = self.args.T
        #
        self.device = torch.device("cuda:0")

    @staticmethod
    def parse_args():
        parser = Attacker.parse_args()
        # parser.add_argument('--selected_ids', type=str, default='1,2,3', required=True)
        #
        parser.add_argument('--restore_model', type=int, default=0)
        parser.add_argument('--model_path', type=str, default='')
        #
        parser.add_argument('--epoch', type=int, default=10)
        parser.add_argument('--batch_size', type=int, default=256)
        #
        parser.add_argument('--learning_rate_G', type=float, default=0.01)
        parser.add_argument('--reg_rate_G', type=float, default=0.0001)
        parser.add_argument('--ZR_ratio', type=float, default=0.2)
        #
        parser.add_argument('--learning_rate_D', type=float, default=0.001)
        parser.add_argument('--reg_rate_D', type=float, default=1e-5)
        #
        parser.add_argument('--verbose', type=int, default=1)
        parser.add_argument('--T', type=int, default=5)
        #
        args, _ = parser.parse_known_args()
        return args

    def prepare_data(self):
        super(AUSH, self).prepare_data()
        train_matrix, _ = self.dataset_class.dataFrame_to_matrix(self.train_data_df, self.n_users, self.n_items)
        self.train_data_array = train_matrix.toarray()
        # let the rated inter be 1 and none be 0
        self.train_data_mask_array = scipy.sign(self.train_data_array)
        # true/false and to the float 1. / 0.
        mask_array = (self.train_data_array > 0).astype(np.float)
        # let selected items and target items be 0
        mask_array[:, self.selected_ids + [self.target_id]] = 0
        self.template_idxs = np.where(np.sum(mask_array, 1) >= self.filler_num)[0]

    def build_network(self):
        self.netG = AushGenerator(input_dim=self.n_items).to(self.device)
        self.G_optimizer = optim.Adam(self.netG.parameters(), lr=self.learning_rate_G)

        self.netD = AushDiscriminator(input_dim=self.n_items).to(self.device)
        self.D_optimizer = optim.Adam(self.netD.parameters(), lr=self.learning_rate_D)

        pass

    def sample_fillers(self, real_profiles):
        fillers = np.zeros_like(real_profiles)
        filler_pool = set(range(self.n_items)) - set(self.selected_ids) - {self.target_id}
        # filler_sampler = lambda x: np.random.choice(x[0], size=x[1], replace=False)
        # sampled_cols = [filler_sampler([filler_pool, self.filler_num]) for _ in range(real_profiles.shape[0])]

        filler_sampler = lambda x: np.random.choice(size=self.filler_num, replace=False,
                                                    a=list(set(np.argwhere(x > 0).flatten()) & filler_pool))
        sampled_cols = [filler_sampler(x) for x in real_profiles]

        sampled_rows = np.repeat(np.arange(real_profiles.shape[0]), self.filler_num)
        fillers[sampled_rows, np.array(sampled_cols).flatten()] = 1
        return fillers

    def train(self):

        # save
        total_batch = math.ceil(len(self.template_idxs) / self.batch_size)
        idxs = np.random.permutation(self.template_idxs)  # shuffled ordering
        #
        g_loss_rec_l = []
        g_loss_shilling_l = []
        g_loss_gan_l = []

        d_loss_list, g_loss_list = [], []
        for i in range(total_batch):

            # ---------------------
            #  Prepare Input
            # ---------------------
            batch_set_idx = idxs[i * self.batch_size: (i + 1) * self.batch_size]

            # Adversarial ground truths
            valid_labels = np.ones_like(batch_set_idx)
            fake_labels = np.zeros_like(batch_set_idx)
            valid_labels = torch.tensor(valid_labels).type(torch.float).to(self.device).reshape(len(batch_set_idx), 1)
            fake_labels = torch.tensor(fake_labels).type(torch.float).to(self.device).reshape(len(batch_set_idx), 1)
            # print(valid_labels)


            # Select a random batch of real_profiles
            real_profiles = self.train_data_array[batch_set_idx, :]
            # sample fillers
            fillers_mask = self.sample_fillers(real_profiles)
            # selected
            selects_mask = np.zeros_like(fillers_mask)
            selects_mask[:, self.selected_ids] = 1.
            # target
            target_patch = np.zeros_like(fillers_mask)
            target_patch[:, self.selected_ids] = 5.
            # ZR_mask
            ZR_mask = (real_profiles == 0) * selects_mask
            pools = np.argwhere(ZR_mask)
            np.random.shuffle(pools)
            pools = pools[:math.floor(len(pools) * (1 - self.ZR_ratio))]
            ZR_mask[pools[:, 0], pools[:, 1]] = 0

            # ----------- torch.mul ---------
            real_profiles = torch.tensor(real_profiles).type(torch.float).to(self.device)
            fillers_mask = torch.tensor(fillers_mask).type(torch.float).to(self.device)
            selects_mask = torch.tensor(selects_mask).type(torch.float).to(self.device)
            target_patch = torch.tensor(target_patch).type(torch.float).to(self.device)
            ZR_mask = torch.tensor(ZR_mask).type(torch.float).to(self.device)
            input_template = torch.mul(real_profiles, fillers_mask)
            # ----------generate----------
            self.netG.eval()
            gen_output = self.netG(input_template)
            gen_output = gen_output.detach()
            # ---------mask--------
            selected_patch = torch.mul(gen_output, selects_mask)
            middle = torch.add(input_template, selected_patch)
            fake_profiles = torch.add(middle, target_patch)
            # --------Discriminator------
            # forward
            self.D_optimizer.zero_grad()
            self.netD.train()
            d_valid_labels = self.netD(real_profiles * (fillers_mask + selects_mask))
            d_fake_labels = self.netD(fake_profiles * (fillers_mask + selects_mask))
            # loss
            print(d_valid_labels.shape)
            # d_valid_labels = d_valid_labels.reshape(-1)
            # d_fake_labels = d_fake_labels.reshape(-1)
            D_real_loss = nn.BCELoss()(d_valid_labels, valid_labels)
            D_fake_loss = nn.BCELoss()(d_fake_labels, fake_labels)
            d_loss = 0.5 * (D_real_loss + D_fake_loss)
            print("d_loss")
            print(d_loss)
            d_loss.backward()
            self.D_optimizer.step()
            self.netD.eval()

            # ---------train G-------
            self.netG.train()
            d_fake_labels = self.netD(fake_profiles * (fillers_mask + selects_mask))
            g_loss_gan = nn.BCELoss()(d_fake_labels, valid_labels)
            g_loss_shilling = nn.MSELoss()(fake_profiles * selects_mask, selects_mask * 5.)
            # g_loss_shilling = (fake_profiles * selects_mask - selects_mask * 5.) ** 2
            # * selects_mask - selects_mask * input_template) * ZR_mask
            # g_loss_rec = (fake_profiles * selects_mask - selects_mask * input_template) * ZR_mask ** 2
            g_loss_rec = nn.MSELoss()(fake_profiles * selects_mask * ZR_mask, selects_mask * input_template * ZR_mask)
            g_loss = g_loss_gan + g_loss_rec + g_loss_shilling
            # g_loss = g_loss_rec + g_loss_shilling
            # + g_loss_shilling + g_loss_gan
            g_loss_rec_l.append(g_loss_rec.item())
            g_loss_shilling_l.append(g_loss_shilling.item())
            g_loss_gan_l.append(g_loss_gan.item())
            self.G_optimizer.zero_grad()
            g_loss.backward(retain_graph=True)
            self.G_optimizer.step()
        print(g_loss_rec_l)
        print(g_loss_shilling_l)
        print(g_loss_gan_l)
        return

    def execute(self):

        self.prepare_data()

        # Build and compile GAN Network
        self.build_network()

        if self.restore_model:
            self.restore(self.model_path)
            print("loading done.")

        else:
            for epoch in range(self.epochs):
                print("epoch %d"%epoch)
                self.train()

                '''if self.verbose and epoch % self.T == 0:
                    print("epoch:%d\td_loss:%.4f\tg_loss:%.4f" % (epoch,g_loss_cur))'''

            # self.save(self.model_path)
            print("training done.")

        metrics = self.test(victim='SVD', detect=True)
        # print(metrics, flush=True)
        return

    def generate_fakeMatrix(self):
        # Select a random batch of real_profiles
        idx = self.template_idxs[np.random.randint(0, len(self.template_idxs), self.attack_num)]
        real_profiles = self.train_data_array[idx, :]
        # sample fillers
        fillers_mask = self.sample_fillers(real_profiles)
        # selected
        selects_mask = np.zeros_like(fillers_mask)
        selects_mask[:, self.selected_ids] = 1.
        # target
        target_patch = np.zeros_like(fillers_mask)
        target_patch[:, self.target_id] = 5.

        # Generate
        real_profiles = torch.tensor(real_profiles).type(torch.float).to(self.device)
        fillers_mask = torch.tensor(fillers_mask).type(torch.float).to(self.device)
        selects_mask = torch.tensor(selects_mask).type(torch.float).to(self.device)
        target_patch = torch.tensor(target_patch).type(torch.float).to(self.device)
        input_template = torch.mul(real_profiles, fillers_mask)
        self.netG.eval()
        gen_output = self.netG(input_template)
        selected_patch = torch.mul(gen_output, selects_mask)
        middle = torch.add(input_template, selected_patch)
        fake_profiles = torch.add(middle, target_patch)
        fake_profiles = fake_profiles.detach().cpu().numpy()
        # fake_profiles = self.generator.predict([real_profiles, fillers_mask, selects_mask, target_patch])
        # selected patches
        selected_patches = fake_profiles[:, self.selected_ids]
        selected_patches = np.round(selected_patches)
        selected_patches[selected_patches > 5] = 5
        selected_patches[selected_patches < 1] = 1
        fake_profiles[:, self.selected_ids] = selected_patches

        return fake_profiles

    def generate_injectedFile(self, fake_array):
        super(AUSH, self).generate_injectedFile(fake_array)
