import shutil

import cv2
from dateutil.parser import parse
from queue import Queue
from tqdm import tqdm
from typing import Tuple, Generator
from datetime import datetime, timedelta
from pathlib import Path
from video import NormalVideoCapture
from PIL import Image
import pytesseract
import numpy as np
from matplotlib import pyplot as plt
from skimage.color import rgb2gray

VIDEO_PATH = Path(__file__, '../../incoming_to_cut').resolve()
CLEAR_TO_CUT = VIDEO_PATH / 'clear_to_cut'
VIDEO_ICP = CLEAR_TO_CUT / 'video_icp'
AUTO_CUT = CLEAR_TO_CUT / 'auto_cut'

def parse_video_file(file: Path) -> Tuple[datetime, int, int]:
    """
    Derives meta info from video filename

    :param file: The path to the file
    :type file: Path
    :return: The timestamp of when the recording started, the bed number and the FPS count
    :rtype: Tuple[datetime, int, int]
    """
    assert file.suffix == '.ogv', f"Unexpected video file type: {file.suffix}"
    date_str, bed, fps = file.stem.rsplit('-', maxsplit=2)
    return parse(date_str), int(bed), int(fps[1:])

def gen_all_videos(path: Path) -> Generator[Tuple[Path, datetime, int, int], None, None]:
    if path.is_dir():
        for f in path.glob('*.ogv'):
            yield f, *parse_video_file(f)
    elif path.is_file():
        yield path, *parse_video_file(path)
    else:
        raise Exception(f"Given path is neither a directory nor a file: {str(path)}")

def find_changes(vcap: cv2.VideoCapture, out_folder: Path, date: datetime, bed: int, fps: float, target_fps: int = 15, queue_padding_in_seconds: timedelta = timedelta(seconds=10.0)):
    backSubKNN = cv2.createBackgroundSubtractorKNN(history=15, dist2Threshold=800.0, detectShadows=False)
    def has_significant_change(frame: np.ndarray) -> bool:
        w, h, _ = frame.shape
        total_pixels = w * h
        fgMaskKNN = backSubKNN.apply(frame, learningRate=0.7)
        changed_pixel_cnt = float(np.count_nonzero(fgMaskKNN))
        # print(v.time_read, changed_pixel_cnt/total_pixels)
        return changed_pixel_cnt/total_pixels > 0.01  # bigger than 2 percent => change detected
    orig_fps = vcap.get(cv2.CAP_PROP_FPS)
    size = tuple(map(int, (vcap.get(cv2.CAP_PROP_FRAME_WIDTH), vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    assert int(orig_fps) == fps, "Video file indicated wrong FPS"
    v = NormalVideoCapture(vcap, target_fps)
    pad = (queue_padding_in_seconds.seconds * target_fps)
    backlog = Queue(maxsize=pad)
    first_frame = True
    # vw = cv2.VideoWriter(str(out_folder / '00.avi'), fourcc=cv2.VideoWriter_fourcc(*'MJPG'), fps=target_fps, frameSize=size)
    # for _ in range(target_fps * 10):
    #     s, f = v.read()
    #     assert s
    #     vw.write(f)
    # vw.release()
    with tqdm(total=v.frames_left) as bar:
        while True:
            # Search for changes for a new clip
            s, f = v.read()
            if not s:
                bar.write("Finished")
                break
            bar.update()
            if backlog.full():
                backlog.get()
            backlog.put(f)
            if not has_significant_change(f) or first_frame:
                first_frame = False
                continue
            first_frame = False
            start_frame_idx = v.frames_read - 1
            timestamp_indicator = v.time_read

            # Write clip to file
            start_offset = max(timestamp_indicator - queue_padding_in_seconds, timedelta(0))
            t = date + start_offset
            bar.write(f" => T+{str(start_offset)} [frame {v.frames_read - 1}]: Found significant changes in video at {str(timestamp_indicator)} [frame {start_frame_idx}], starting from {str(t)}")
            filename = out_folder / f"{t.year}-{t.month:02}-{t.day:02}-{t.hour:02}{t.minute:02}{t.second:02}-{bed}-r{target_fps}.avi"
            vw = cv2.VideoWriter(str(filename), fourcc=cv2.VideoWriter_fourcc(*'MJPG'), fps=target_fps, frameSize=size)
            while not backlog.empty():  # Write backlogged frames to file
                vw.write(backlog.get())
            back_pad = 0
            while back_pad < pad:
                back_pad += 1
                s, f = v.read()
                bar.update()
                if not s:
                    bar.write("EMPTY FRAME READ!!!")
                    break
                vw.write(f)
                if backlog.full():
                    backlog.get()
                backlog.put(f)
                if has_significant_change(f):  # We're not done yet. Extend the clip
                    back_pad = 0
            vw.release()
            frames_count = v.frames_read - start_frame_idx
            duration2 = timedelta(seconds=frames_count / v.target_fps)
            duration = v.time_read - start_offset
            new_filename = filename.with_name(f"{filename.stem}-d{int(duration.total_seconds()*100):06}.avi")
            shutil.move(filename, new_filename)
            bar.write(f"  ==> T+{str(start_offset + duration)} [frame {v.frames_read}]: Created {str(duration)} [{frames_count} frames] long clip ({str(t)} until {str(t + duration)}): {new_filename.name}")



if __name__ == '__main__':
    AUTO_CUT.mkdir(parents=True, exist_ok=True)
    for f, date, bed, fps in gen_all_videos(VIDEO_ICP):
        # f = f.with_name('00.ogv')
        cap: cv2.VideoCapture = cv2.VideoCapture(str(f))
        # v = NormalVideoCapture(cap, 1)
        # frames = []
        # s = True
        # with tqdm(total=v.frames_left) as bar:
        #     while s:
        #         bar.update()
        #         s, f = v.read()
        #         frames.append(f)
        print(f.name, str(date), bed)
        find_changes(cap, AUTO_CUT, date, bed, fps)
