import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
# filter import
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter
# Cola import
from ColaLib import draw_box

# ready data
fp = open("./ColaDataset_tmp.pkl", 'rb')
img_list, pred_list, pred_cof = pickle.load(fp)
fp.close()

def trans_Y(pred_list):
    l = pred_list.__len__()
    Y = np.zeros((4, l), dtype=np.float32)
    for i in range(l):
        Y[:, i] = pred_list[i].numpy()
    return Y
Y_senser = trans_Y(pred_list)

class ColaKalmanFilter(KalmanFilter):
    def __init__(self, dim_x, dim_z, dt, dim_u=0):
        super().__init__(dim_x, dim_z, dim_u)
        self.dt = dt
        
        # self.x = None
        self.B = 0
        
        # F: nx X nx
        self.F = np.zeros((dim_x, dim_x))
        tmp = np.array([
            [1, dt, 0.5*dt**2], 
            [0, 1, dt], 
            [0, 0, 1]
        ])
        self.F[0:3, 0:3] = tmp
        self.F[3:6, 3:6] = tmp
        self.F[6:9, 6:9] = tmp
        self.F[9:12, 9:12] = tmp
        
        # H: nz X nx
        self.H = np.zeros((dim_z, dim_x))
        for i in range(dim_z):
            self.H[i, i * 3] = 1
            
        # P: nx X nx
        tmp = np.tile(np.array([1, 20, 20]), 4)
        self.P = np.diag(tmp)
        
        # Q: nx X nx
        sigma_axy = 10
        sigma_aw = 10
        sigma_ar = 0.1
        self.Q = np.zeros((dim_x, dim_x))
        Q_tmp = np.array([
            dt**4/4, dt**3/2, dt**2/2, 
            dt**3/2, dt**2, dt, 
            dt**2/2, dt, 1, 
        ]).reshape(3, 3)
        self.Q[0:3, 0:3] = Q_tmp * sigma_axy
        self.Q[3:6, 3:6] = Q_tmp * sigma_axy
        self.Q[6:9, 6:9] = Q_tmp * sigma_aw
        self.Q[9:12, 9:12] = Q_tmp * sigma_ar
        
        # P: nz X nz
        self.R = np.diag([50, 50, 20, 0.1])
        
    
    def get_next_steps_z_hat(self, n: int):
        """
        return: next n steps z_hat, n = 1, 2, 3, ..., self.dt
                shape=(dim_z, n)
        """
        dt = 1
        F = np.zeros((self.dim_x, self.dim_x))
        tmp = np.array([
            [1, dt, 0.5*dt**2], 
            [0, 1, dt], 
            [0, 0, 1]
        ])
        F[0:3, 0:3] = tmp
        F[3:6, 3:6] = tmp
        F[6:9, 6:9] = tmp
        F[9:12, 9:12] = tmp
        
        z_tensor = np.zeros((self.dim_z, n), dtype=np.float32)
        x = self.x.copy()
        for i in range(n):
            x = np.dot(F, x)
            z_tensor[:, i] = self.measurement_of_state(x).reshape(-1)
        
        return self._z_to_box(z_tensor)
        
    
    def get_box(self):
        """
        return: z_hat, shape(dim_z, 1), H * X
        """
        z_hat = self.measurement_of_state(self.x)
        return self._z_to_box(z_hat)
        
        
        
    def init_x(self, z_0):
        """
        z_0: first measure data, ndarray, shape(4, 1)
        """
        z_0 = z_0.reshape(4, 1)
        z_0 = self._box_to_z(z_0)
        self.x = np.column_stack((z_0.reshape(4, -1), np.zeros((4, 2)))).reshape(-1, 1)
        
    
    def _box_to_z(self, box):
        """
        z: ndarray shape(4, N), z[3, :] -> h
        """
        box[3, :] = box[3, :] / (box[2, :])
        return box
    

    def _z_to_box(self, z):
        """
        z: shape(4, N), z[3, :] -> r
        """
        z[3, :] = z[3, :] * z[2, :]
        return z
    
        
    def cola_update(self, box):
        """
        z: ndarray shape=(4, 1), z[3]->w_h
        """
        # ready z
        if box is None:
            z = self.measurement_of_state(self.x)
        else:
            box = box.astype(np.float32).reshape(self.dim_z, -1)
            dis_min = 1e10
            box_min = None
            z_hat = self.measurement_of_state(self.x)
            z = None
            for idx_box in range(box.shape[1]):
                z_ = self._box_to_z(box[:, idx_box].reshape(4, 1))
                dis = np.linalg.norm(z_hat - z_)
                print("dis = ", dis)
                if dis < dis_min:
                    z = z_
                    dis_min = dis
            
            # print(np.linalg.norm(z))
            print("z = ", z)
            print("z_hat = ", z_hat)
            print("z - z_hat = ", z - z_hat)
            if np.linalg.norm(z - z_hat) > np.linalg.norm(z_hat) * 0.8:
                z = z_hat
        z[z < 0] = 0

        self.update(z)
    
        

class ColaKalmanFilter_v1:
    def __init__(self, main_widget=None) -> None:
        self.main_widget = main_widget
        
        # Parameters
        self.step, self.X, self.P, self.H, self.Q, self.F, self.R = \
            self.get_parameters()
            
            
    def get_curr_Y(self):
        return self.H @ self.X
    
    
    def init_paramters(self, step, first_y0):
        self.step, self.X, self.P, self.H, self.Q, self.F, self.R = \
            self.get_parameters(step)
        if first_y0 is not None:
            self.init_X_with_first_snap(first_y0)
    

    def get_parameters(self, step=10):
        X = np.zeros((8, 1), dtype=np.float32)
        P = np.eye(8, dtype=np.float32) * 10
        H = np.column_stack(
            (np.eye(4), np.zeros((4, 4)))
        ).astype(np.float32)
        Q = 0.0001 * np.eye(8, dtype=np.float32)
        R = np.diag([1, 1, 10, 10]).astype(np.float32)
        F = np.eye(8).astype(np.float32)
        F[0:4, 4:8] = step * np.eye(4)
        return step, X, P, H, Q, F, R


    def init_X_with_first_snap(self, Y0):
        """
        Y0: ndarray 
        """
        self.X = np.asarray([
            Y0.reshape(-1),   
            [0, 0, 0, 0]], 
            dtype=np.float32).reshape(-1, 1)
    
    
    def update(self, Y):
        """
        Y: ndarray (8, 1)
        """
        Y = Y.reshape(-1, 1)
        # prediction
        X_hat = self.F @ self.X 
        P_hat = self.F @ self.P @ self.F.T + self.Q
        
        # some intermediate
        S = self.H @ P_hat @ self.H.T + self.R
        K = P_hat @ self.H.T @ np.linalg.inv(S)

        # upate
        self.X = X_hat + K @ (Y - self.H @ X_hat)
        self.P = (np.eye(8) - K @ self.H) @ P_hat

        # show result
        Y_update = self.H @ self.X
        return Y_update


    def predict_next(self, step_num):
        """return the prediction box in next step_num steps

        Args:
            step_num(int): <= self.step
        """
        boxs = np.zeros((4, step_num), dtype=np.float32)
        F = np.eye(8).astype(np.float32)
        F[0:4, 4:8] = np.eye(4)
        X = self.X
        for i in range(step_num):
            X = F @ X
            boxs[:, i] = (self.H @ X).reshape(-1)
        return boxs
        

def draw_circle_with_boxsize(img, box, color=(0, 0, 255)):
    raidus = max(int(min(box[2:4]) / 10), 1)
    img = cv2.circle(img, (int(box[0]), int(box[1])), raidus, color, -1)
    # return img


def main():
    cola_kalmanfilter = ColaKalmanFilter_v1()
    for i in range(1, len(pred_list)):
        img = img_list[i]
        box = cola_kalmanfilter.update(pred_list[i].numpy()).reshape(-1).tolist()
        # prediction
        boxs = cola_kalmanfilter.predict_next(10)
        for i in range(boxs.shape[1]):
            draw_circle_with_boxsize(img, boxs[:, i])
        
        img = draw_box(img, box)
        cv2.imshow("WIN", img)
        cv2.waitKey(30)
    
    
def main2():
    f = ColaKalmanFilter(12, 4, dt=5)
    f.init_x(pred_list[0])
    for i in range(1, len(pred_list)):
        img = img_list[i]
                
        # measurement
        box = pred_list[i].numpy().reshape(-1).astype(np.float32)
        # Add Process noise
        p_noise = np.random.randn(4) * np.array([5, 5, 5, 0.1])
        box = box + p_noise
        # Out of sight
        print(i)
        # if i in [3, 4]:
        #     box = None
        
        # Far away!
        # if i in [6]:
        #     box = np.ones((4, 1), dtype=np.float32)
        
        # prediction
        f.predict()
        
        # update
        f.cola_update(box)

        # do something with z_hat
        box = f.get_box().reshape(-1)
        img = draw_box(img, box, color=(0, 0, 255))
        print(box)
        boxs = f.get_next_steps_z_hat(5)
        for i in range(boxs.shape[1]):
            draw_circle_with_boxsize(img, boxs[:, i])
        
        cv2.imshow("WIN", img)
        cv2.waitKey(0)
        

if __name__ == '__main__':
    main2()