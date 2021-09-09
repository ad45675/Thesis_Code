
def LSF(Data,NowPos):

    for i in range(4):
        if i<4:
            Data[i]=Data[i+1]
        elif(i==4):
            Data[i]=NowPos

    Vel=Data[0]*0.3+Data[1]*0.1+Data[2]*0.1+Data[3]*0.3

    return Vel
