import numpy as np
import math


def FindOptSol(invsol, nowjointpos,Singularplace):
    sol = invsol
    issolution = np.zeros((8, 1), np.int)
    mindistance = 0
    buffer = 0
    num_of_sol = 0
    optimalsol = np.zeros((6,), np.float)

    for i in range(8):
        if (abs(invsol[i][3]) < 3.0 and abs(invsol[i][0]<1.6) and Singularplace[i] == 1):
            issolution[i] = 1
        else:
            issolution[i] = 0


    for i in range(8):
        for j in range(6):
            if abs(sol[i][j]) >= 3.14500:
                issolution[i] = 0
        if issolution[i] == 1:
            num_of_sol = num_of_sol + 1


    if num_of_sol == 0:
        print('此點在工作範圍外,無解無解')
        optimalsol[0] = 0.0
        optimalsol[1] = 0.0
        optimalsol[2] = 0.0
        optimalsol[3] = 0.0
        optimalsol[4] = 0.0
        optimalsol[5] = 0.0
    else:
        optimalsol_num = 0
        for i in range(8):
            if (issolution[i] == 1):
                distance = 0
                for j in range(6):
                    buffer = invsol[i][j] - nowjointpos[j]
                    distance = distance + buffer * buffer

                    buffer = 0
                if (optimalsol_num == 0):
                    mindistance = distance
                    optimalsol_num = i+1
                else:
                    if (distance < mindistance):
                        mindistance = distance
                        optimalsol_num = i+1

                # print('dis',i, distance)


        optimalsol[0] = sol[optimalsol_num-1][0]
        optimalsol[1] = sol[optimalsol_num-1][1]
        optimalsol[2] = sol[optimalsol_num-1][2]
        optimalsol[3] = sol[optimalsol_num-1][3]
        optimalsol[4] = sol[optimalsol_num-1][4]
        optimalsol[5] = sol[optimalsol_num-1][5]
    return optimalsol
