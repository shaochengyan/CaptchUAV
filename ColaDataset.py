import pickle
import cv2

class ColaDataset:
    def __init__(self):
        self.img_list = []  # orignal image
        self.pred_list = []  # result in this image
        self.pred_conf = []
        
        # record img index
        self.len = 0
        self.curr_idx = -1

        # parameters
        self.max_frame_num = 10
        self.video_interval_frame = 5
        
    
    def save_data(self):
        fp = open("./ColaDataset_tmp.pkl", 'wb')
        pickle.dump((self.img_list, self.pred_list, self.pred_conf), fp)
        fp.close()
    
        
    def init_paramenter(self, max_frame_num, video_inter):
        self.max_frame_num = max_frame_num
        self.video_interval_frame = video_inter
        
        
    def update_idx(self, step: int):
        self.curr_idx += step
        
        if self.curr_idx >= self.len:
            self.curr_idx = self.len - 1
        elif self.curr_idx < 0:
            self.curr_idx = 0
            
        
    
    def get_curr_data(self):
        return self[self.curr_idx]
    
    
    def append(self, img, pred):
        self.img_list.append(img)
        self.pred_list.append(pred)
        
    
    def update(self, filename_list, filetype):
        if filetype == 'VideoFile(*.mp4)':
            cap = cv2.VideoCapture(filename_list[0])
            num = self.max_frame_num
            while num > 0:
                num -= 1
                ret, img = cap.read()
                i = self.video_interval_frame
                while i > 0:
                    i -= 1
                    cap.read()
                    
                if not ret:
                    break
                self.img_list.append(img)
                self.pred_list.append(None)
                self.pred_conf.append(None)
        elif filetype == 'ImageFile(*.jpg;*.png)':
            for filename in filename_list:
                img = cv2.imread(filename)
                self.img_list.append(img)
                self.pred_list.append(None)
                self.pred_conf.append(None)
        self.len = len(self)
        self.curr_idx = 0
    
    
    def clear(self):
        # todo: clearn object like this, will it still store in memory?
        while len(self.img_list):
            del self.img_list[0]
            del self.pred_list[0]
        self.img_list = []
        self.pred_list = []
        self.len = 0
        self.curr_idx = -1
    
    
    def __iter__(self):
        return zip(self.img_list, self.pred_list)
    
    
    def __getitem__(self, idx):
        if not( idx >= 0 and idx < self.len):
            raise(Exception("Index out of range"))
        self.curr_idx = idx
        return self.img_list[idx].copy(), self.pred_list[idx]
    
    
    def __len__(self):
        return len(self.img_list)