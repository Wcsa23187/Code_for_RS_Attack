# use this code not only you can get the best value you also can get the best choice if the item 
w, n = 100, 134
# 12
# 2 1 0
weight = weight
value = v
p = [[0 for j in range(w + 1)] for i in range(n)]
rec = [[0 for j in range(w + 1)] for i in range(n)]

for i in range(1, n):
    for j in range(w + 1):
        if weight[i] <= j and value[i] + p[i - 1][j - weight[i]] > p[i - 1][j]:
            p[i][j] = value[i] + p[i - 1][j - weight[i]]
            rec[i][j] = 1
        else:
            p[i][j] = p[i - 1][j]
print(p[n - 1][w])
print("choose item ", end="")
tmp = w
for i in range(n - 1, -1, -1):
    if rec[i][tmp] == 1:
        print((v[i], weight[i], user_id[i], i), end=" ")

        tmp -= weight[i]
