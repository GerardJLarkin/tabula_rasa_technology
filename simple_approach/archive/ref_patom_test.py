# ref.py
import numpy as np

class RefPatom:
    def __init__(self, patom: np.ndarray):
        self.count = 1
        # store just the second‐row summary
        self.second = patom[1:2].astype(np.float64)
        # build histograms for x, y, colour
        pts = patom[2:,:3]
        # self.x_hist = np.bincount(pts[:,0].astype(int))
        # self.y_hist = np.bincount(pts[:,1].astype(int))
        x_counts, x_bin_edges = np.histogram( pts[:,0], bins=64, range=(-1,1) )
        self.x_hist = x_counts; print('x_counts', x_counts)
        y_counts, y_bin_edges = np.histogram( pts[:,1], bins=64, range=(-1,1) )
        self.y_hist = y_counts


        self.c_sum = pts[:,2].sum()
        self.n_pts = len(pts)

    def update(self, patom: np.ndarray):
        pts = patom[2:,:3]
        x = pts[:,0].astype(int); y = pts[:,1].astype(int); c = pts[:,2]
        # update histograms (fast C code)
        self.x_hist = np.bincount(
            np.concatenate([np.arange(len(self.x_hist)), x]),
            minlength=max(len(self.x_hist), x.max()+1)
        )
        self.y_hist = np.bincount(
            np.concatenate([np.arange(len(self.y_hist)), y]),
            minlength=max(len(self.y_hist), y.max()+1)
        )
        self.c_sum += c.sum()
        self.n_pts += len(c)
        # running mean of second row
        self.second = (self.second*self.count + patom[1:2]) / (self.count+1)
        self.count += 1

    def reference(self) -> np.ndarray:
        # reconstruct fixed‐size patom <=64×64 by sampling the top‐n bins
        n = int(np.ceil(self.n_pts / self.count)); print('FOOOOOOK', n)
        top_x = np.argsort(self.x_hist); print(top_x.shape)
        top_y = np.argsort(self.y_hist); print(top_y.shape)
        colours = np.full(n, self.c_sum/self.n_pts, dtype=np.float32); print(colours.shape)
        pts = np.column_stack([top_x, top_y, colours])
        return np.vstack([
            [np.nan, np.nan, np.nan, np.nan],
            self.second.astype(np.float32),
            pts.astype(np.float32)
        ])
