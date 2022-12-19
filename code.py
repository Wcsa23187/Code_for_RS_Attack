 alpha = float(sys.argv[sys.argv.index('--alpha') + 1])
        a = int(alpha * self.attack_num)
        b = int(self.attack_num - a)
        print(a)
        print(b)

        user = [5520, 5678, 1771, 4738, 317, 4962, 1338, 4975, 970, 3305, 5, 646, 1802, 2191, 2704, 3987, 789, 3734,
                5017, 797, 286, 160, 4770, 35, 2340, 2597, 425, 2347, 5682, 4918, 1462, 3386, 1082, 2365, 4926, 3903,
                587, 3532, 4434, 1238, 4572, 5854, 5345, 1509, 362, 4330, 623, 5618, 2678, 3582]
        dict_user_code = {}
        j = 0
        for i in user:
            dict_user_code[i] = j
            j+=1
        user_po = [4926, 4962, 5618, 623, 587, 3582, 4770, 4572, 4975, 3386, 2191, 2704, 4738, 286, 5017, 5345, 1238, 2365,
                    789, 317, 3305, 425, 1338, 797, 1771,]
        user_ne = [ 35, 970, 5, 2678, 2347, 3903, 3987, 4918, 4434, 5678, 5682, 1082,
                    4330, 2597, 362, 3734, 1509, 1802, 160, 3532, 1462, 2340, 646, 5520, 5854]
        x1 = np.random.choice(user_po,a)
        x2 = np.random.choice(user_ne,b)
        x= list(x1) + list(x2)
        sampled_idx = []
        for i in x:
            sampled_idx.append(dict_user_code[i])
        print(sampled_idx)
        target_user_array_path = './data/ml1m/target_user_train.npy'
        target_user_array = np.load(target_user_array_path)
        print(self.train_array[sampled_idx].shape)
        print(target_user_array.shape)
        templates = target_user_array[sampled_idx]
        print(templates.shape)
