import numpy as np

DH = np.array([
    [0, 0.675, 0.26, -np.pi/2],
    [0, 0, 0.68, 0],
    [0, 0, 0.035, -np.pi/2],
    [0, 0.67, 0, np.pi/2],
    [0, 0, 0, -np.pi/2],
    [0, 0.12, 0, 0]
])

def fkine(q):
    T = np.eye(4)
    for i in range(6):
        theta = q[i] + DH[i,0]
        d     = DH[i,1]
        a     = DH[i,2]
        alpha = DH[i,3]

        T_i = np.array([
            [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
            [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
            [0,              np.sin(alpha),                np.cos(alpha),               d],
            [0,              0,                            0,                           1]
        ])
        T = T @ T_i
    pos = T[:3,3]
    R = T[:3,:3]
    pitch = np.arctan2(-R[2,0], np.sqrt(R[0,0]**2 + R[1,0]**2))
    roll  = np.arctan2(R[2,1]/np.cos(pitch), R[2,2]/np.cos(pitch))
    yaw   = np.arctan2(R[1,0]/np.cos(pitch), R[0,0]/np.cos(pitch))
    return np.array([pos[0], pos[1], pos[2], roll, pitch, yaw])
