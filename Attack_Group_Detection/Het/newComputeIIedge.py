import Parser
import Bulidata
import time
from tqdm import tqdm
import getData
import networkx
import datetime
import dateutil
import math
import scipy.stats
from dateutil.relativedelta import relativedelta
import sys



def date_minus_month(gk_date):
    d = datetime.datetime.strptime(str(gk_date), "%Y-%m-%d")
    date_mm = d + relativedelta(months=1)
    date_mm = datetime.datetime.strftime(date_mm, "%Y-%m-%d")
    return date_mm

def intersect(a, b):
    return list(set(a) & set(b))



def time_del(time1, time2, type="day"):
    """
    计算时间差
    :param time1: 较小的时间（datetime类型）
    :param time2: 较大的时间（datetime类型）
    :param type: 返回结果的时间类型（暂时就是返回相差天数）
    :return: 相差的天数
    """
    day1 = time.strptime(str(time1), '%Y-%m-%d')
    day2 = time.strptime(str(time2), '%Y-%m-%d')
    if type == 'day':
        day_num = (int(time.mktime(day2)) - int(time.mktime(day1))) / (
            24 * 60 * 60)
    return abs(int(day_num))

def JS_divergence(p,q):
    M=(p+q)/2
    return 0.5*scipy.stats.entropy(p, M)+0.5*scipy.stats.entropy(q, M)


if __name__ == '__main__':
    args = Parser.Define_Params()
    file_path = '.' + args.file_path

    UItoRT, Times, UserToItem, ItemToUser, newUItoRT = getData.bulid_dataset(file_path)

    startTime = min(Times)
    endTime = max(Times)
    startTime = str(startTime)[0:10]
    endTime = str(endTime)[0:10]
    iu_neigh = args.iu_neigh

    TimeSeque = []
    while startTime <= endTime:
        startTime = date_minus_month(startTime)
        TimeSeque.append(startTime)
    print(TimeSeque)
    iiindexes = []
    time_thresold = 20
    rat_thresold = 2
    finish = 0
    file_path = './edges/iiedge1.txt'
    for user in UserToItem.keys():
        sitem = len(UserToItem[user])
        finishlen = len(UserToItem.keys())
        finish = finish + 1
        if finish % 100 == 0:
            print('the all finishlen is %d , now run the %dth' % (finishlen, finish))
        for i in range(sitem):
            for j in range(i + 1, sitem):
                iiindex = 'i' + UserToItem[user][i] + 'i' + UserToItem[user][j]
                if iiindex not in iiindexes:
                    iiindexes.append(iiindex)
                else:
                    continue
                item1 = UserToItem[user][i]
                item2 = UserToItem[user][j]
                users1 = ItemToUser[item1]
                users2 = ItemToUser[item2]
                common_users = intersect(users1, users2)
                bing_users = list(set(users1).union(set(users2)))
                temp = 0
                for single_user in common_users:
                    uiindex1 = 'u' + str(single_user) + 'i' + str(item1)
                    uiindex2 = 'u' + str(single_user) + 'i' + str(item2)
                    rat1 = int(newUItoRT[uiindex1][0])
                    rat2 = int(newUItoRT[uiindex2][0])
                    time1 = newUItoRT[uiindex1][1]
                    time2 = newUItoRT[uiindex2][1]
                    days = time_del(time1, time2)
                    rats = math.fabs((rat1 - rat2))

                    if days < time_thresold:
                        time_ratio = days/time_thresold
                    else:
                        time_ratio = 0
                    if rats < rat_thresold:
                        rat_ratio = rats / rat_thresold
                    else:
                        rat_ratio = 0
                    des = (1 - time_ratio) / 2 + (1 - rat_ratio) / 2
                    temp = temp + des
                corres = (len(common_users) / len(bing_users)) * temp

                with open(file_path, 'a') as file:
                    file.write(str(iiindex) + ' ' + str(corres) + '\n')











