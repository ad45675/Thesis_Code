import  math
import numpy as np

DegToRad = math.pi/ 180
RadToDeg = 180/math.pi

DH_table = np.array([[0,            0.345,  0.08,   math.pi / 2],
					 [0+math.pi / 2 , 0,  0.27,     0],
					 [0,             0,     0.09,    math.pi / 2],
					 [0,            0.295,  0,       -math.pi / 2],
					 [0,            0,      0,       math.pi / 2],
					 [0,       0.102+0.06835, 0,          0]])
#
# jointangle=[0,30*DegToRad,-20*DegToRad,0,-45*DegToRad,0]
def compute_robot_jacobian(DH_table,jointangle):


    s1 = math.sin(jointangle[0] + DH_table[0][0])
    s2 = math.sin(jointangle[1] + DH_table[1][0])
    s3 = math.sin(jointangle[2] + DH_table[2][0])
    s4 = math.sin(jointangle[3] + DH_table[3][0])
    s5 = math.sin(jointangle[4] + DH_table[4][0])
    s6 = math.sin(jointangle[5] + DH_table[5][0])

    c1 = math.cos(jointangle[0] + DH_table[0][0])
    c2 = math.cos(jointangle[1] + DH_table[1][0])
    c3 = math.cos(jointangle[2] + DH_table[2][0])
    c4 = math.cos(jointangle[3] + DH_table[3][0])
    c5 = math.cos(jointangle[4] + DH_table[4][0])
    c6 = math.cos(jointangle[5] + DH_table[5][0])

    d1 = DH_table[0][1]
    d2 = DH_table[1][1]
    d3 = DH_table[2][1]
    d4 = DH_table[3][1]
    d5 = DH_table[4][1]
    d6 = DH_table[5][1]

    a1 = DH_table[0][2]
    a2 = DH_table[1][2]
    a3 = DH_table[2][2]
    a4 = DH_table[3][2]
    a5 = DH_table[4][2]
    a6 = DH_table[5][2]

    Jacobian =np.array([
    [a3 * s1 * s2 * s3 - d6 * (c5 * (c2 * s1 * s3 + c3 * s1 * s2) - s5 * (c1 * s4 - c4 * (c2 * c3 * s1 - s1 * s2 * s3))) - d4 * ( c2 * s1 * s3 + c3 * s1 * s2) - a2 * c2 * s1 - a1 * s1 - a3 * c2 * c3 * s1, -c1 * ( a2 * s2 - d4 * (c2 * c3 - s2 * s3) - d6 * (c5 * (c2 * c3 - s2 * s3) - c4 * s5 * (c2 * s3 + c3 * s2)) + a3 * c2 * s3 + a3 * c3 * s2), c1 * ( d4 * (c2 * c3 - s2 * s3) + d6 * (c5 * (c2 * c3 - s2 * s3) - c4 * s5 * (c2 * s3 + c3 * s2)) - a3 * c2 * s3 - a3 * c3 * s2), (d6 * ( c5 * (c2 * s1 * s3 + c3 * s1 * s2) - s5 * (c1 * s4 - c4 * (c2 * c3 * s1 - s1 * s2 * s3))) + d4 * (c2 * s1 * s3 + c3 * s1 * s2)) * ( c2 * c3 - s2 * s3) - (d4 * (c2 * c3 - s2 * s3) + d6 * (c5 * (c2 * c3 - s2 * s3) - c4 * s5 * (c2 * s3 + c3 * s2))) * (c2 * s1 * s3 + c3 * s1 * s2), d6 * (c1 * c4 + s4 * (c2 * c3 * s1 - s1 * s2 * s3)) * (c5 * (c2 * c3 - s2 * s3) - c4 * s5 * (c2 * s3 + c3 * s2)) + d6 * s4 * (c5 * (c2 * s1 * s3 + c3 * s1 * s2) - s5 * ( c1 * s4 - c4 * (c2 * c3 * s1 - s1 * s2 * s3))) * (c2 * s3 + c3 * s2), 0],
    [a1 * c1 + d6 * (c5 * (c1 * c2 * s3 + c1 * c3 * s2) + s5 * (s1 * s4 + c4 * (c1 * c2 * c3 - c1 * s2 * s3))) + d4 * (c1 * c2 * s3 + c1 * c3 * s2) + a2 * c1 * c2 - a3 * c1 * s2 * s3 + a3 * c1 * c2 * c3, -s1 * (a2 * s2 - d4 * (c2 * c3 - s2 * s3) - d6 * (c5 * (c2 * c3 - s2 * s3) - c4 * s5 * (c2 * s3 + c3 * s2)) + a3 * c2 * s3 + a3 * c3 * s2), s1 * (d4 * (c2 * c3 - s2 * s3) + d6 * ( c5 * (c2 * c3 - s2 * s3) - c4 * s5 * (c2 * s3 + c3 * s2)) - a3 * c2 * s3 - a3 * c3 * s2), (c1 * c2 * s3 + c1 * c3 * s2) * (d4 * (c2 * c3 - s2 * s3) + d6 * (c5 * (c2 * c3 - s2 * s3) - c4 * s5 * (c2 * s3 + c3 * s2))) - ( c2 * c3 - s2 * s3) * (d6 * (c5 * (c1 * c2 * s3 + c1 * c3 * s2) + s5 * (s1 * s4 + c4 * (c1 * c2 * c3 - c1 * s2 * s3))) + d4 * (c1 * c2 * s3 + c1 * c3 * s2)), d6 * (c5 * (c2 * c3 - s2 * s3) - c4 * s5 * (c2 * s3 + c3 * s2)) * (c4 * s1 - s4 * (c1 * c2 * c3 - c1 * s2 * s3)) - d6 * s4 * (c5 * (c1 * c2 * s3 + c1 * c3 * s2) + s5 * (s1 * s4 + c4 * (c1 * c2 * c3 - c1 * s2 * s3))) * (c2 * s3 + c3 * s2), 0],
    [0, c1 * (d6 * (c5 * (c1 * c2 * s3 + c1 * c3 * s2) + s5 * (s1 * s4 + c4 * (c1 * c2 * c3 - c1 * s2 * s3))) + d4 * ( c1 * c2 * s3 + c1 * c3 * s2) + a2 * c1 * c2 - a3 * c1 * s2 * s3 + a3 * c1 * c2 * c3) + s1 * (d6 * (c5 * (c2 * s1 * s3 + c3 * s1 * s2) - s5 * (c1 * s4 - c4 * (c2 * c3 * s1 - s1 * s2 * s3))) + d4 * (c2 * s1 * s3 + c3 * s1 * s2) + a2 * c2 * s1 - a3 * s1 * s2 * s3 + a3 * c2 * c3 * s1), s1 * (d6 * (c5 * (c2 * s1 * s3 + c3 * s1 * s2) - s5 * (c1 * s4 - c4 * (c2 * c3 * s1 - s1 * s2 * s3))) + d4 * (c2 * s1 * s3 + c3 * s1 * s2) - a3 * s1 * s2 * s3 + a3 * c2 * c3 * s1) + c1 * ( d6 * (c5 * (c1 * c2 * s3 + c1 * c3 * s2) + s5 * (s1 * s4 + c4 * (c1 * c2 * c3 - c1 * s2 * s3))) + d4 * (c1 * c2 * s3 + c1 * c3 * s2) - a3 * c1 * s2 * s3 + a3 * c1 * c2 * c3), (d6 * (c5 * (c2 * s1 * s3 + c3 * s1 * s2) - s5 * (c1 * s4 - c4 * (c2 * c3 * s1 - s1 * s2 * s3))) + d4 * (c2 * s1 * s3 + c3 * s1 * s2)) * (c1 * c2 * s3 + c1 * c3 * s2) - (c2 * s1 * s3 + c3 * s1 * s2) * (d6 * (c5 * (c1 * c2 * s3 + c1 * c3 * s2) + s5 * (s1 * s4 + c4 * (c1 * c2 * c3 - c1 * s2 * s3))) + d4 * (c1 * c2 * s3 + c1 * c3 * s2)), d6 * (c1 * c4 + s4 * (c2 * c3 * s1 - s1 * s2 * s3)) * (c5 * (c1 * c2 * s3 + c1 * c3 * s2) + s5 * (s1 * s4 + c4 * (c1 * c2 * c3 - c1 * s2 * s3))) + d6 * (c5 * (c2 * s1 * s3 + c3 * s1 * s2) - s5 * (c1 * s4 - c4 * (c2 * c3 * s1 - s1 * s2 * s3))) * ( c4 * s1 - s4 * (c1 * c2 * c3 - c1 * s2 * s3)), 0],
    [0, s1, s1, c1 * c2 * s3 + c1 * c3 * s2, c4 * s1 - s4 * (c1 * c2 * c3 - c1 * s2 * s3), c5 * (c1 * c2 * s3 + c1 * c3 * s2) + s5 * (s1 * s4 + c4 * (c1 * c2 * c3 - c1 * s2 * s3))],
    [0, -c1, -c1, c2 * s1 * s3 + c3 * s1 * s2, - c1 * c4 - s4 * (c2 * c3 * s1 - s1 * s2 * s3), c5 * (c2 * s1 * s3 + c3 * s1 * s2) - s5 * (c1 * s4 - c4 * (c2 * c3 * s1 - s1 * s2 * s3))],
    [1, 0, 0, s2 * s3 - c2 * c3, -s4 * (c2 * s3 + c3 * s2), c4 * s5 * (c2 * s3 + c3 * s2) - c5 * (c2 * c3 - s2 * s3)]
    ])


    #只有位置的jacobian

    Jv = np.array([
        [d6 * (s5 * (c1 * s4 + c4 * (s1 * s2 * s3 - c2 * c3 * s1)) - c5 * (c2 * s1 * s3 + c3 * s1 * s2)) - d4 * ( c2 * s1 * s3 + c3 * s1 * s2) - a1 * s1 - a2 * c2 * s1 - a3 * c2 * c3 * s1 + a3 * s1 * s2 * s3, - d4 * (c1 * s2 * s3 - c1 * c2 * c3) - d6 * (c5 * (c1 * s2 * s3 - c1 * c2 * c3) + c4 * s5 * (c1 * c2 * s3 + c1 * c3 * s2)) - a2 * c1 * s2 - a3 * c1 * c2 * s3 - a3 * c1 * c3 * s2, - d4 * (c1 * s2 * s3 - c1 * c2 * c3) - d6 * (c5 * (c1 * s2 * s3 - c1 * c2 * c3) + c4 * s5 * (c1 * c2 * s3 + c1 * c3 * s2)) - a3 * c1 * c2 * s3 - a3 * c1 * c3 * s2,d6 * s5 * (c4 * s1 + s4 * (c1 * s2 * s3 - c1 * c2 * c3)),d6 * (c5 * (s1 * s4 - c4 * (c1 * s2 * s3 - c1 * c2 * c3)) - s5 * (c1 * c2 * s3 + c1 * c3 * s2)), 0],
        [d6 * (s5 * (s1 * s4 - c4 * (c1 * s2 * s3 - c1 * c2 * c3)) + c5 * (c1 * c2 * s3 + c1 * c3 * s2)) + d4 * (c1 * c2 * s3 + c1 * c3 * s2) + a1 * c1 + a2 * c1 * c2 + a3 * c1 * c2 * c3 - a3 * c1 * s2 * s3, - d4 * (s1 * s2 * s3 - c2 * c3 * s1) - d6 * (c5 * (s1 * s2 * s3 - c2 * c3 * s1) + c4 * s5 * ( c2 * s1 * s3 + c3 * s1 * s2)) - a2 * s1 * s2 - a3 * c2 * s1 * s3 - a3 * c3 * s1 * s2, - d4 * (s1 * s2 * s3 - c2 * c3 * s1) - d6 * (c5 * (s1 * s2 * s3 - c2 * c3 * s1) + c4 * s5 * ( c2 * s1 * s3 + c3 * s1 * s2)) - a3 * c2 * s1 * s3 - a3 * c3 * s1 * s2, -d6 * s5 * ( c1 * c4 - s4 * (s1 * s2 * s3 - c2 * c3 * s1)), -d6 * (c5 * (c1 * s4 + c4 * (s1 * s2 * s3 - c2 * c3 * s1)) + s5 * (c2 * s1 * s3 + c3 * s1 * s2)), 0],
        [0, d4 * (c2 * s3 + c3 * s2) + a2 * c2 + d6 * ( c5 * (c2 * s3 + c3 * s2) + c4 * s5 * (c2 * c3 - s2 * s3)) + a3 * c2 * c3 - a3 * s2 * s3, d4 * (c2 * s3 + c3 * s2) + d6 * (c5 * (c2 * s3 + c3 * s2) + c4 * s5 * ( c2 * c3 - s2 * s3)) + a3 * c2 * c3 - a3 * s2 * s3, -d6 * s4 * s5 * (c2 * s3 + c3 * s2), d6 * (s5 * (c2 * c3 - s2 * s3) + c4 * c5 * (c2 * s3 + c3 * s2)), 0]
        ])

    return Jacobian


# #test jacobian
# jointangle = [0,0,0,0,0,0]
# Jacobian=compute_robot_jacobian(DH_table,jointangle)
# b=np.linalg.det(Jacobian)
# print(b)


