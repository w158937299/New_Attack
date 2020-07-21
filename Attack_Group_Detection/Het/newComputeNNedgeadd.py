


iu_neigh = {}
uu_neigh = {}
ui_neigh = {}
ii_neigh = {}
uu_edge_weight = {}
ii_edge_weight = {}


format_uu_edge_filepath = './edges/format_UUedge1.txt'
format_ii_edge_filepath = './edges/format_IIedge1.txt'

uu_neigh_filepath = './edges/uu_neigh1.txt'
ii_neigh_filepath = './edges/ii_neigh1.txt'

ui_neigh_filepath = './edges/ui_neigh.txt'
iu_neigh_filepath = './edges/iu_neigh.txt'



with open(iu_neigh_filepath, 'r') as file:
    lines = file.readlines()
    for line in lines:

        line = line.strip('\n')
        line = line.split('  ')
        item = 'i' + line[0]
        temp = []
        Users = line[1][1:-1].split(',')
        for user in Users:
            des_user = 'u' + user.strip()[1:-1]
            temp.append(des_user)

        iu_neigh.setdefault(item, temp)

with open(uu_neigh_filepath, 'r') as file:
    lines = file.readlines()
    for line in lines:
        line = line.strip('\n')
        line = line.split('   ')
        uu_neigh.setdefault(line[0], line[1])

#
with open(ui_neigh_filepath, 'r') as file:
    lines = file.readlines()
    for line in lines:
        line = line.strip('\n')
        line = line.split('   ')
        ui_neigh.setdefault(line[0], line[1])

with open(ii_neigh_filepath, 'r') as file:
    lines = file.readlines()
    for line in lines:
        line = line.strip('\n')
        line = line.split('   ')
        ii_neigh.setdefault(line[0], line[1])

for curnode in uu_neigh:
    temp = uu_neigh[curnode][1:-1].split(',')
    an = []
    for user in temp:
        des_user = user.strip()[1:-1]
        an.append(des_user)
    uu_neigh.setdefault(curnode, list(an))


for curnode in ui_neigh:
    temp = ui_neigh[curnode][1:-1].split(',')
    an = []
    for user in temp:
        des_user = user.strip()[1:-1]
        an.append(des_user)
    ui_neigh.setdefault(curnode, list(an))

for curnode in ii_neigh:
    temp = ii_neigh[curnode][1:-1].split(',')
    an = []
    for user in temp:
        des_user = user.strip()[1:-1]
        an.append(des_user)
    ii_neigh.setdefault(curnode, list(an))

needed_user = list(set(ui_neigh.keys()).difference(set(uu_neigh.keys())))
needed_item = list(set(iu_neigh.keys()).difference(set(ii_neigh.keys())))

add_uu_neigh = {}
add_ii_neigh = {}

for user in needed_user:
    for i in uu_neigh:
        if user in uu_neigh[i]:
            if user not in add_uu_neigh:
                add_uu_neigh.setdefault(user, [i])
            else:
                temp = add_uu_neigh[user]
                temp.append(i)
                add_uu_neigh[user] = temp

for user in needed_item:
    for i in ii_neigh:
        if user in ii_neigh[i]:
            if user not in add_ii_neigh:
                add_ii_neigh.setdefault(user, [i])
            else:
                temp = add_ii_neigh[user]
                temp.append(i)
                add_ii_neigh[user] = temp

with open(format_uu_edge_filepath, 'r') as file:
    lines = file.readlines()
    for line in lines:
        line = line.strip('\n')
        line = line.split('  ')
        uu_edge_weight.setdefault(line[0], line[1])

with open(format_ii_edge_filepath, 'r') as file:
    lines = file.readlines()
    for line in lines:
        line = line.strip('\n')
        line = line.split('  ')
        ii_edge_weight.setdefault(line[0], line[1])





# add()


uukeys = []
iikeys = []
Fileter_uu_neigh = {}
Fileter_ii_neigh = {}

# 有些权重计算中确实没有
for user in add_uu_neigh:
    for user1 in add_uu_neigh[user]:
        uu = user1 + user
        uu1 = user + user1
        if uu in uu_edge_weight or uu1 in uu_edge_weight:
            if user not in Fileter_uu_neigh:
                Fileter_uu_neigh.setdefault(user, [user1])
            else:
                temp = Fileter_uu_neigh[user]
                temp.append(user1)
                Fileter_uu_neigh[user] = temp

for user in add_ii_neigh:
    for user1 in add_ii_neigh[user]:
        uu = user1 + user
        uu1 = user + user1
        if uu in ii_edge_weight or uu1 in ii_edge_weight:
            if user not in Fileter_ii_neigh:
                Fileter_ii_neigh.setdefault(user, [user1])
            else:
                temp = Fileter_ii_neigh[user]
                temp.append(user1)
                Fileter_ii_neigh[user] = temp

with open(uu_neigh_filepath, 'a') as file:
    for i in Fileter_uu_neigh:
        file.write(str(i) + '   ' + str(Fileter_uu_neigh[i]) + '\n')

with open(ii_neigh_filepath, 'a') as file:
    for i in Fileter_ii_neigh:
        file.write(str(i) + '   ' + str(Fileter_ii_neigh[i]) + '\n')

#
# print(uukeys)
# print(iikeys)




















