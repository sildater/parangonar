import numpy as np
import scipy
from enum import IntEnum
from queue import Queue
from scipy.spatial.distance import cdist

def element_of_metric(vec1, vec2):
    """
    metric that evaluates occurence of vec2 (scalar) in vec1 (vector n-dim)
    """
    return 2 - np.sum(vec2 == vec1)

def cdist_local(arr1, arr2, metric):
    """
    compute array of pairwise distances between 
    the elements of two arrays given a metric
    
    Parameters
    ----------
    arr1: numpy nd array
    
    arr2: numpy nd array
    
    metric: callable
        a metric function
    
    Returns
    -------
    
    pdist_array: numpy 2d array
        array of pairwise distances
    """
    pdist_array = np.ones((arr1.shape[0],arr2.shape[0]))*np.inf
    for i in range(arr1.shape[0]):
        for j in range(arr2.shape[0]):
            pdist_array[i, j] = metric(arr1[i], arr2[j])
    return pdist_array

class Direction(IntEnum):
    REF = 0
    INPUT = 1
    BOTH = 2

    def toggle(self):
        return Direction(self ^ 1) if self != Direction.BOTH else Direction.INPUT

class OLTW(object):
    """
    On-line Dynamic Time Warping (OLTW) algorithm for aligning a input sequence to a reference sequence.
    Adapted from: https://github.com/laurenceyoon/python-match

    Parameters
    ----------
    reference_features : np.ndarray
        A 2D array with dimensions (`T(n_timesteps)`, `F(n_features)`) containing the
        features of the reference the input is going to be aligned to.

    queue : Queue
        A queue for storing the input features, which shares the same object as the audio stream.

    window_size : int
        Size of the window (in steps) of the cost matrix.

    max_run_count : int
        Maximum number of times the class can run in the same direction.

    hop_size : int
        number of seq items that get added at step

    directional_weights: np.ndarray
        weights associated with each of the three possible steps

    Attributes
    ----------
    warping_path : np.ndarray [shape=(2, T)]
        Warping path with pairs of indices of the reference and input features.
        where warping_path[0] is the index of the reference feature and 
        warping_path[1] is the index of the input feature.
    """

    def __init__(
        self,
        reference_features = None,
        input_feature_shape = None,
        queue = None,
        window_size= 10, # shape of the acc cost matric
        max_run_count = 100, # maximal number of steps 
        hop_size = 1, # number of seq items that get added at step
        directional_weights = np.array([1, 1, 1]), # down, diag, right
        cdist_fun = cdist,
        cdist_metric = "euclidean",
        **kwargs,
    ):
        self.queue = queue
        self.cdist_fun = cdist_fun
        self.cdist_metric = cdist_metric
        self.directional_weights = directional_weights
        if reference_features is not None:
            self.set_feature_arrays(reference_features, 
                                    input_feature_shape)
        else:
            self.reference_features = None
            self.N_ref = None
            self.input_features = None
        self.w = window_size
        self.max_run_count = max_run_count
        self.hop_size = hop_size
        self.initialize()

    def set_feature_arrays(self, 
                        reference_features,
                        input_feature_shape = None,
                        ):
        self.reference_features = reference_features
        self.N_ref = self.reference_features.shape[0]
        if input_feature_shape is not None:
            self.input_features = np.empty((0, input_feature_shape))
        else:
            self.input_features = np.empty((0, self.reference_features.shape[1]))

    def initialize(self):
        self.ref_pointer = 0
        self.input_pointer = 0
        self.run_count = 0
        self.previous_direction = None
        self.wp = np.array([[-1, -1]]).T  # [shape=(2, T)]

        self.ref_pointer += self.w # window of ref shifted at start
        self.acc_dist_matrix = np.full((self.w, self.w), np.inf)
        self.acc_dist_matrix[0,:] = 0
        self.acc_len_matrix = np.zeros((self.w, self.w))
        self.queue_non_empty = True
        self.local_both_dist = np.zeros((self.w, self.w))

    @property
    def warping_path(self):  # [shape=(2, T)]
        return self.wp[:,1:]

    def offset(self):
        offset_x = max(self.ref_pointer - self.w, 0)
        offset_y = max(self.input_pointer - self.w, 0)
        return np.array([offset_x, offset_y])

    def update_ref_direction(self,
                              dist, 
                              new_acc, 
                              new_len_acc, 
                              wx, 
                              wy, 
                              d):
        update_x0 = wx - d
        for i in range(d):
            for j in range(wy):
                local_dist = dist[i, j]
                if j == 0:
                    new_acc[update_x0 + i, j] = local_dist * self.directional_weights[0] + new_acc[update_x0 + i - 1, j]
                    new_len_acc[update_x0 + i, j] = 1 + new_len_acc[update_x0 + i - 1, j]
                else:
                    compares = [
                        new_acc[update_x0 + i - 1, j] + local_dist * self.directional_weights[0], 
                        new_acc[update_x0 + i, j - 1] + local_dist * self.directional_weights[2] ,
                        new_acc[update_x0 + i - 1, j - 1] + local_dist * self.directional_weights[1], # diagonal
                    ] 
                    len_compares = [
                        new_len_acc[update_x0 + i - 1, j],
                        new_len_acc[update_x0 + i, j - 1],
                        new_len_acc[update_x0 + i - 1, j - 1],
                    ]
                    local_direction = np.argmin(compares)
                    new_acc[update_x0 + i, j] =  compares[local_direction]
                    new_len_acc[update_x0 + i, j] = 1 + len_compares[local_direction]
        return new_acc, new_len_acc

    def update_input_direction(self, 
                                dist, 
                                new_acc, 
                                new_len_acc, 
                                wx, 
                                wy, 
                                d):
        update_y0 = wy - d
        for i in range(wx):
            for j in range(d):
                local_dist = dist[i, j]
                if i == 0:
                    new_acc[i, update_y0 + j] = local_dist * self.directional_weights[2] + new_acc[i, update_y0 - 1 + j]
                    new_len_acc[i, update_y0 + j] = 1 + new_len_acc[i, update_y0 - 1 + j]
                else:
                    compares = [
                        new_acc[i - 1, update_y0 + j] + local_dist * self.directional_weights[0],
                        new_acc[i, update_y0 + j - 1] + local_dist * self.directional_weights[2],
                        new_acc[i - 1, update_y0 + j - 1] + local_dist * self.directional_weights[1], # diagonal
                    ]
                    len_compares = [
                        new_len_acc[i - 1, update_y0 + j],
                        new_len_acc[i, update_y0 + j - 1],
                        new_len_acc[i - 1, update_y0 + j - 1],
                    ]
                    local_direction = np.argmin(compares)
                    new_acc[i, update_y0 + j] = compares[local_direction]
                    new_len_acc[i, update_y0 + j] = 1 + len_compares[local_direction]
        return new_acc, new_len_acc

    def update_both_direction(self, 
                                dist, 
                                new_acc, 
                                new_len_acc, 
                                wx, 
                                wy, 
                                d):
        for i in range(wx):
            for j in range(wy):
                local_dist = dist[i, j]
                if i == 0 and j >= wy - d:
                    new_acc[i, j] = local_dist * self.directional_weights[2] + new_acc[i, j - 1]
                    new_len_acc[i, j] = 1 + new_len_acc[i, j - 1]
                elif i >= wx - d and j == 0:
                    new_acc[i, j] = local_dist * self.directional_weights[0] + new_acc[i - 1, j]
                    new_len_acc[i, j] = 1 + new_len_acc[i - 1, j]
                elif i >= wx - d or j >= wy - d:
                    compares = [
                        new_acc[i - 1, j] + local_dist * self.directional_weights[0],
                        new_acc[i,  j - 1] + local_dist * self.directional_weights[2],
                        new_acc[i - 1, j - 1] + local_dist * self.directional_weights[1],
                    ]
                    len_compares = [
                        new_len_acc[i - 1,  j],
                        new_len_acc[i, j - 1],
                        new_len_acc[i - 1, j - 1],
                    ]
                    local_direction = np.argmin(compares)
                    new_acc[i, j] = compares[local_direction]
                    new_len_acc[i, j] = 1 + len_compares[local_direction]
        return new_acc, new_len_acc
    

    def update_both_direction_new(self, 
                                dist, 
                                new_acc, 
                                new_len_acc, 
                                wx, 
                                wy, 
                                d):
        
        for j in range(wy - d, wy):
            new_acc[0, j] = dist[0, j] * self.directional_weights[2] + new_acc[0, j - 1]
            new_len_acc[0, j] = 1 + new_len_acc[0, j - 1]

        for i in range(wx - d, wx):
            new_acc[i, 0] = dist[i, 0] * self.directional_weights[0] + new_acc[i - 1, 0]
            new_len_acc[i, 0] = 1 + new_len_acc[i - 1, 0]
        
        # first add the columns to the right
        # do not go all the way to the corner
            
        for i in range(1, wx - d): 
            for j in range(wy - d, wy):
                local_dist = dist[i, j]
                compares = [
                    new_acc[i - 1, j] + local_dist * self.directional_weights[0],
                    new_acc[i,  j - 1] + local_dist * self.directional_weights[2],
                    new_acc[i - 1, j - 1] + local_dist * self.directional_weights[1],
                ]
                len_compares = [
                    new_len_acc[i - 1,  j],
                    new_len_acc[i, j - 1],
                    new_len_acc[i - 1, j - 1],
                ]
                local_direction = np.argmin(compares)
                new_acc[i, j] = compares[local_direction]
                new_len_acc[i, j] = 1 + len_compares[local_direction]

        # then add the rows at the bottom
        for i in range(wx - d, wx):
            for j in range(1, wy):
                local_dist = dist[i, j]
                compares = [
                    new_acc[i - 1, j] + local_dist * self.directional_weights[0],
                    new_acc[i,  j - 1] + local_dist * self.directional_weights[2],
                    new_acc[i - 1, j - 1] + local_dist * self.directional_weights[1],
                ]
                len_compares = [
                    new_len_acc[i - 1,  j],
                    new_len_acc[i, j - 1],
                    new_len_acc[i - 1, j - 1],
                ]
                local_direction = np.argmin(compares)
                new_acc[i, j] = compares[local_direction]
                new_len_acc[i, j] = 1 + len_compares[local_direction]

        
        return new_acc, new_len_acc

    def update_cost_matrix(self, direction):
        # local cost matrix
        x, y = self.ref_pointer, self.input_pointer
        wx, wy = min(self.w, x), min(self.w, y)
        d = self.hop_size
        new_acc = np.full((wx, wy), np.inf, dtype=np.float32)
        # new_acc[0,0] = 0
        new_len_acc = np.zeros((wx, wy))

        if direction is Direction.REF: 
            new_acc[:-d, :] = self.acc_dist_matrix[d:]
            new_len_acc[:-d, :] = self.acc_len_matrix[d:]
            x_seg = self.reference_features[x - d : x]  # [d, 12]
            y_seg = self.input_features[y - wy : y]  # [wy, 12]
            dist = self.cdist_fun(x_seg, y_seg, metric=self.cdist_metric)  # [wx, d]
            new_acc, new_len_acc = self.update_ref_direction(
                dist, new_acc, new_len_acc, wx, wy, d)

        elif direction is Direction.INPUT:  
            overlap_y = wy - d
            new_acc[:, :-d] = self.acc_dist_matrix[:, -overlap_y:]
            new_len_acc[:, :-d] = self.acc_len_matrix[:, -overlap_y:]
            x_seg = self.reference_features[x - wx : x]  # [wx, 12]
            y_seg = self.input_features[y - d : y]  # [d, 12]
            dist = self.cdist_fun(x_seg, y_seg, metric=self.cdist_metric)  # [wx, d]
            new_acc, new_len_acc = self.update_input_direction(
                dist, new_acc, new_len_acc, wx, wy, d)
            
        elif direction is Direction.BOTH:  # input, BOTH
            overlap_y = wy - d
            new_acc[:-d, :-d] = self.acc_dist_matrix[d:, -overlap_y:]
            new_len_acc[:-d, :-d] = self.acc_len_matrix[d:, -overlap_y:]

            # old costly distance computation
            # x_seg = self.reference_features[x - wx : x]  # [wx, 12]
            # y_seg = self.input_features[y - wy : y]  # [wy, 12]
            # dist = self.cdist_fun(x_seg, y_seg, metric=self.cdist_metric)  # [wx, d]
            # new, compute only outer slice
            x_seg = self.reference_features[x - wx : x - d]  # [wx - d]
            y_seg = self.input_features[y - d : y]  # [d]
            self.local_both_dist[:wx - d, -d:] = self.cdist_fun(x_seg, y_seg, metric=self.cdist_metric)
            x_seg = self.reference_features[x - d : x]  # [d]
            y_seg = self.input_features[y - wy : y]  # [wy]
            self.local_both_dist[wx - d:, :] = self.cdist_fun(x_seg, y_seg, metric=self.cdist_metric)

            new_acc, new_len_acc = self.update_both_direction_new(
                self.local_both_dist, new_acc, new_len_acc, wx, wy, d)
        self.acc_dist_matrix = new_acc
        self.acc_len_matrix = new_len_acc

    def first_cost_matrix(self):
        # local cost matrix
        x, y = self.ref_pointer, self.input_pointer # should be window, hop_size
        wx, wy = x, y + 1 # append an extra column
        d = self.hop_size
        new_acc = np.full((wx, wy), np.inf, dtype=np.float32)
        new_acc[0,0] = 0 # starting point in corner
        new_len_acc = np.zeros((wx, wy))
        x_seg = self.reference_features[x - wx : x]  # [wx, 12]
        y_seg = self.input_features[y - d : y]  # [d, 12]
        dist = self.cdist_fun(x_seg, y_seg, metric=self.cdist_metric)  # [wx, d]
        new_acc, new_len_acc = self.update_input_direction(
            dist, new_acc, new_len_acc, wx, wy, d)
        
        self.acc_dist_matrix = new_acc[:,1:]
        self.acc_len_matrix = new_len_acc[:,1:]

    def select_candidate(self):
        norm_x_edge = self.acc_dist_matrix[-1, :] / self.acc_len_matrix[-1, :]
        norm_y_edge = self.acc_dist_matrix[:, -1] / self.acc_len_matrix[:, -1]
        cat = np.concatenate((norm_x_edge, norm_y_edge))
        
        min_idx = np.argmin(cat)
        offset = self.offset()
        if min_idx < len(norm_x_edge):
            self.candidate = np.array([self.ref_pointer - offset[0] - 1, min_idx])
        else:
            self.candidate = np.array(
                [min_idx - len(norm_x_edge), self.input_pointer - offset[1] - 1]
            )

        # print("edge",cat)
        # print("pointers:", self.ref_pointer, self.input_pointer, offset)
        # print("cand",self.candidate, min_idx)

    def add_candidate_to_path(self):
        new_coordinate = np.expand_dims(self.offset() + self.candidate, axis=1)  # [2, 1]
        self.wp = np.concatenate((self.wp, new_coordinate), axis=1)

    def select_next_direction(self):
        if self.input_pointer < self.w:
            next_direction = Direction.INPUT
        elif self.run_count > self.max_run_count:
            next_direction = self.previous_direction.toggle()
        else:
            offset = self.offset()
            x0, y0 = offset[0], offset[1]
            ref_bool = self.candidate[0] == self.ref_pointer - x0 - 1
            input_bool = self.candidate[1] == self.input_pointer - y0 - 1
            if ref_bool and input_bool:
                next_direction = Direction.BOTH
            elif ref_bool:
                next_direction = Direction.REF
            else:
                assert input_bool
                next_direction = Direction.INPUT
        return next_direction

    def get_new_input(self):
        try:
            input_feature = self.queue.get(block=False)
            self.input_features = np.vstack([self.input_features, input_feature])
            self.input_pointer += self.hop_size

        except:
            print("empty queue")
            self.queue_non_empty = False

    def is_still_following(self):
        is_still_following = self.ref_pointer <= (self.N_ref - self.hop_size)
        return is_still_following and self.queue_non_empty


    def handle_direction(self, direction):
        # get fitting features/windows
        if direction is not Direction.REF: 
                self.get_new_input()
        if direction is not Direction.INPUT:
            self.ref_pointer += self.hop_size

        # accumulate run count
        if direction == self.previous_direction:
            self.run_count += 1
        else:
            self.run_count = 1
        self.previous_direction = direction
    
    def handle_first_input(self):
        direction = self.select_next_direction()
        self.handle_direction(direction)
        self.first_cost_matrix()
        self.select_candidate()
        self.add_candidate_to_path()

    def run(self, verbose=False):
        print("Start running OLTW")
        self.initialize()
        self.handle_first_input()
        while self.is_still_following():
            direction = self.select_next_direction()
            if verbose: 
                print("ACC DIST \n", self.acc_dist_matrix)
                print("ACC LEN \n", self.acc_len_matrix)
                print("RECENT WARPING PATH \n", self.warping_path[:,-3:])
                print("NEXT DIRECTION \n", direction)
                print("*"*50)
            print("RECENT WARPING PATH: ", self.warping_path[:,-1:])
            self.handle_direction(direction)
            self.update_cost_matrix(direction)
            self.select_candidate()
            self.add_candidate_to_path()
        print("... and we're done.")
        return self.warping_path
    

if __name__ == "__main__":
    RANGE_L = 6
    REPEATS = 3
    HOP_SIZE = 1
    WINDOW_SIZE = 3
    # r = np.row_stack([np.arange(RANGE_L).reshape(-1,1) for k in range(REPEATS)]).astype(float)
    # t = np.row_stack([np.roll(np.arange(RANGE_L),0,0).reshape(-1,1) for k in range(REPEATS)]).astype(float)
    # t +=0.5
    # r = np.arange(40).reshape(-1,4)
    # t = np.arange(40).reshape(-1,1)

    r = np.row_stack([np.arange(3).reshape(-1,3) for k in range(10)])
    t = np.row_stack([np.arange(3).reshape(-1,1) for k in range(10)])   
     
    queue = Queue() 
    # for tt in range((RANGE_L * REPEATS) // HOP_SIZE):
    #     queue.put(t[tt*HOP_SIZE: (tt+1)*HOP_SIZE,:])
    for tt in range(40):
        queue.put(t[tt:tt+1,:])

    o = OLTW(reference_features=r, 
             input_feature_shape = 1,
             queue=queue, 
             frame_per_seg = HOP_SIZE,
             window_size = WINDOW_SIZE,
             max_run_count = 4,
             directional_weights = np.array([1.05, 1.1, 1]),
             cdist_fun=cdist_local,
             cdist_metric = element_of_metric
             
             
             )
    p = o.run()
    print("path \n", p)