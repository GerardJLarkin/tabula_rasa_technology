import platform
import subprocess
import time
from typing import Tuple
import numpy as np

## import packages/libraries
from time import perf_counter, process_time, clock_gettime_ns, CLOCK_REALTIME
from itertools import product
from multiprocessing import Pool, cpu_count
import sqlite3

import datetime

from cProfile import Profile
from pstats import SortKey, Stats
import psutil

import sys
sys.path.append('/home/gerard/Desktop/capstone_project')

from b_snapshot_2d_pattern_v9 import patoms2d
from c_pattern_2d_compare_v9 import pattern_compare_2d

source_db = "ref2d_v5.db"  # Change this to your actual database file
memory_db = sqlite3.connect(":memory:")  # Create an in-memory DB

# Attach to disk database and copy all data into memory
disk_conn = sqlite3.connect(source_db)
disk_conn.backup(memory_db)  # Copies entire DB into memory
disk_conn.close()  # Close disk connection, work only in memory

cursor = memory_db.cursor()

# read 2d ref database into memory and convert to ref patoms for comparison
working_ref_patoms = []
ref_names = [name for (name,) in cursor.execute("select name from sqlite_master where type='table' and name like 'ref%';").fetchall()][:20]
for i in ref_names:
    table = cursor.execute(f"select * from {i};").fetchall()
    table_rows = []
    for row in table:
        table_row = [np.frombuffer(elem, dtype=np.float32) for elem in row]
        table_rows.append(table_row)
    table_array = np.array(table_rows)[:,:,-1]
    working_ref_patoms.append(table_array)

ref_patoms_array = np.vstack(working_ref_patoms).astype('float32')
ref_indices = np.unique(ref_patoms_array[:,7],axis=0)

# Function to measure total CPU time of all processes
def get_total_cpu_time():
    process = psutil.Process()  # Get the main process
    total_time = process.cpu_times().user  # Get CPU time of main process
    for child in process.children(recursive=True):  # Add child process CPU time
        total_time += child.cpu_times().user
    return total_time


class VideoStreamFFmpeg:
    def __init__(self, src: int, fps: int, resolution: Tuple[int, int]):
        self.src = src
        self.fps = fps
        self.resolution = resolution
        self.pipe = self._open_ffmpeg()
        self.frame_shape = (self.resolution[1], self.resolution[0], 3)
        self.frame_size = np.prod(self.frame_shape)
        self.wait_for_cam()

    def _open_ffmpeg(self):

        command = [
            'ffmpeg',
            '-f', "v4l2",
            "-input_format",
            "mjpeg",
            '-r', str(self.fps),
            '-video_size', f'{self.resolution[0]}x{self.resolution[1]}',
            '-i', f"{self.src}",
            '-vcodec', 'mjpeg',  # Input codec set to mjpeg
            '-an', '-vcodec', 'rawvideo',  # Decode the MJPEG stream to raw video
            '-pix_fmt', 'bgr24',
            '-vsync', '2',
            '-f', 'image2pipe', '-'
        ]

        return subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8
        )

    def read(self):
        raw_image = self.pipe.stdout.read(self.frame_size)
        if len(raw_image) != self.frame_size:
            return None
        image = np.frombuffer(raw_image, dtype=np.uint8).reshape(self.frame_shape)
        return image

    def release(self):
        self.pipe.terminate()

    def wait_for_cam(self):
        for _ in range(30):
            frame = self.read()
        if frame is not None:
            return True
        return False



def main():
    fsp = 30
    resolution = (240, 320)
    ff_cam = VideoStreamFFmpeg(src="/dev/video0", fps=fsp, resolution=resolution)
    # for run_task in [False, True]:
    seq_ind = 0
    while seq_ind < 30:
        frame = ff_cam.read()
        frame = (frame[..., 0] << 16) | (frame[..., 1] << 8) | frame[..., 2]
        frame = (frame - frame.min()) / (frame.max() - frame.min())
        # print(frame.shape)
        frame_patoms = patoms2d(240, 320, frame, seq_ind)
        print(len(frame_patoms))

        # num_patoms = len(frame_patoms)
        # # print(num_patoms)
        # ############## SECOND TASK: COMPARE NEW PATOMS AGAINST REF PATOMS ###############
        # atime = perf_counter(), process_time()
        # start_cpu = get_total_cpu_time()
        # with Pool(processes=8) as pool:
        #     indices = list(product(range(num_patoms), range(len(working_ref_patoms))))
        #     items = [(frame_patoms[i[0]], working_ref_patoms[i[1]]) for i in indices]
        #     comp_results = pool.starmap(pattern_compare_2d, items) #e.g. ['pcol','px','py','pxc','pyc','pq','pqlen','pfind','xc_d','yc_d','x_d','y_d','rfind','similar']
        #     end_cpu = get_total_cpu_time()
        #     print("Real Time to compare 2D patterns with multiprocessing (secs):", (perf_counter()-atime[0]))
        #     print("CPU Time to compare 2D patterns with multiprocessing (secs):", (process_time()-atime[1]))
        #     print(f"Total CPU Time (all processes): {end_cpu - start_cpu:.8f} sec")

        seq_ind += 1


if __name__ == "__main__":
    start = datetime.datetime.now()
    print(start)
    main()
    end = datetime.datetime.now()
    print(end, 'diff:', end - start)







# import ffmpeg
# import logging
# import time
# logger = logging.getLogger(__name__)

# width = 640
# height = 480
# fps = 30

# reader = (ffmpeg.input('/dev/video0', framerate=fps)
#     .output('pipe:', format='rawvideo', pix_fmt='rgb24')
#     .run_async(pipe_stdout=True))
    
# while True:
#     start = time.time_ns()
#     frame = reader.stdout.read(width * height * 3)
#     print(type(frame))
#     logger.debug(f'frame read time: {(time.time_ns() - start)/1e6}ms')
#     # Do something else
#     logger.debug(f'frame capture loop: {(time.time_ns() - start)/1e6}ms')