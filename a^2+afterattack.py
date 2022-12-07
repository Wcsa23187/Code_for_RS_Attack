import numpy as np

# load the training data and let them to be the matrix 
train_dict = np.load(r'F:\RSlib\data_latest\ml1m\matrix\training_dict.npy', allow_pickle=True).item()
user_num = 5950
item_num = 3702
a = np.zeros((user_num, item_num))
for i in train_dict.keys():
    for j in train_dict[i]:
        a[i][j] = 1

# load the A^2 matrix and get the 50 groups data 
num_2_matrix = np.load('F:\RSlib\data_latest\ml1m\matrix\data_2_matrix.npy', allow_pickle=True)
num_2_matrix = num_2_matrix.item()
matrix_50 = num_2_matrix[49]
print(matrix_50.shape)

# we will compute the sum of each colmuns and row then sort them , we call it the A^2 index
user_dict = {}
for i in range(5950):
    cnt = 0
    for j in range(3702):
        cnt += matrix_50[i][j]
    user_dict[i] = cnt

user_sorted = sorted(user_dict.items(),
                     key=lambda x: x[1], reverse=True)
user_list = [i[0] for i in user_sorted]

item_dict = {}
for j in range(3702):
    cnt = 0
    for i in range(5950):
        cnt += matrix_50[i][j]
    item_dict[j] = cnt

item_sorted = sorted(item_dict.items(),
                     key=lambda x: x[1], reverse=True)

item_list = [i[0] for i in item_sorted]

# a2_index : 0--> max user_id   0---> max item_id
a2_index = np.zeros((user_num, item_num))
for i in range(5950):
    for j in range(3702):
        a2_index[i][j] = matrix_50[user_list[i]][item_list[j]]

'''corr = a2_index
sns.heatmap(corr, cmap="Blues")
plt.title('show')
plt.show()'''

# 3664 * 1064
train_dict = np.load(r'F:\RSlib\data_latest\ml1m\matrix\training_dict.npy', allow_pickle=True).item()
user_num = 5950
item_num = 3702


train = np.zeros((user_num, item_num))
for i in range(5950):
    for j in range(3702):
        user = user_list[i]
        item = item_list[j]
        if item in train_dict[user]:
            train[i][j] = 1

corr = train
sns.heatmap(corr, cmap="Blues")
plt.title('show')
plt.show()



# show the graph of the recommendation list 10 with the A^2 index
light_10 = np.load('F:\RSlib\data_latest\ml1m\matrix\lgn_10_matrix.npy')
light_index = np.zeros((user_num,item_num))
for i in range(5950):
    for j in range(3702):
        light_index[i][j] = light_10[user_list[i]][item_list[j]]
corr = light_index
sns.heatmap(corr, cmap="Blues")
plt.title('show')
plt.show()

# change the view you can let the recom matrix as the index

light_10 = np.load('F:\RSlib\data_latest\ml1m\matrix\lgn_10_matrix.npy')
user_num = 5950
item_num = 3702
light_index = np.zeros((user_num,item_num))

user_dict = {}
for i in range(5950):
    cnt = 0
    for j in range(3702):
        cnt += light_10[i][j]
    user_dict[i] = cnt

user_sorted = sorted(user_dict.items(),
                     key=lambda x: x[1], reverse=True)
user_list = [i[0] for i in user_sorted]

item_dict = {}
for j in range(3702):
    cnt = 0
    for i in range(5950):
        cnt += light_10[i][j]
    item_dict[j] = cnt

item_sorted = sorted(item_dict.items(),
                     key=lambda x: x[1], reverse=True)

item_list = [i[0] for i in item_sorted]

'''
num_2_matrix = np.load('F:\RSlib\data_latest\ml1m\matrix\data_2_matrix.npy', allow_pickle=True)
num_2_matrix = num_2_matrix.item()
matrix_50 = num_2_matrix[49]
a2_index = np.zeros((user_num, item_num))
for i in range(5950):
    for j in range(3702):
        a2_index[i][j] = matrix_50[user_list[i]][item_list[j]]
'''


a2_index = np.zeros((user_num, item_num))
for i in range(5950):
    for j in range(3702):
        a2_index[i][j] = light_10[user_list[i]][item_list[j]]

corr = a2_index
sns.heatmap(corr, cmap="Blues")
plt.title('show')
plt.show()
