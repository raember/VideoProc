import shutil

import cv2
import numpy as np
from datetime import datetime, timedelta
from dateutil.parser import parse
from pathlib import Path
from queue import Queue
from tqdm import tqdm
from typing import Tuple, Generator

from video import NormalVideoCapture

VIDEO_PATH = Path(__file__, '../../incoming_to_cut').resolve()
CLEAR_TO_CUT = VIDEO_PATH / 'clear_to_cut'
VIDEO_ICP = CLEAR_TO_CUT / 'video_icp'
AUTO_CUT = CLEAR_TO_CUT / 'auto_cut'
MIN_SIGNIFICANT_FRAME_LENGTH = timedelta(seconds=5)

def parse_video_file(file: Path) -> Tuple[datetime, int, int]:
    """
    Derives meta info from video filename

    :param file: The path to the file
    :type file: Path
    :return: The timestamp of when the recording started, the bed number and the FPS count
    :rtype: Tuple[datetime, int, int]
    """
    date_str, bed, fps = file.stem.rsplit('-', maxsplit=2)
    return parse(date_str), int(bed), int(fps[1:])

def parse_video_clip_file(file: Path) -> Tuple[datetime, int, int, timedelta]:
    """
    Derives meta info from video clip filename

    :param file: The path to the file
    :type file: Path
    :return: The timestamp of when the clip started, the bed number, the FPS count and the duration of the clip
    :rtype: Tuple[datetime, int, int, timedelta]
    """
    date_str, bed, fps, dur = file.stem.rsplit('-', maxsplit=3)
    return parse(date_str), int(bed), int(fps[1:]), timedelta(seconds=float(dur[1:])/100)

def gen_all_videos(path: Path) -> Generator[Tuple[Path, datetime, int, int], None, None]:
    if path.is_dir():
        for f in path.glob('*.ogv'):
            yield f, *parse_video_file(f)
    elif path.is_file():
        yield path, *parse_video_file(path)
    else:
        raise Exception(f"Given path is neither a directory nor a file: {str(path)}")

def find_changes(vcap: cv2.VideoCapture, out_folder: Path, date: datetime, bed: int, fps: float, target_fps: int = 15, queue_padding_in_seconds: timedelta = timedelta(seconds=10.0), black_out_timestamps: bool = False):
    def has_significant_change(frame: np.ndarray, last_frame: np.ndarray) -> bool:
        w, h, _ = frame.shape
        total_pixels = w * h
        current_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        previous_frame_gray = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
        frame_diff = cv2.absdiff(current_frame_gray, previous_frame_gray)
        changed_pixel_cnt = np.count_nonzero(frame_diff > 5)
        #print(v.time_read, changed_pixel_cnt/total_pixels)
        return changed_pixel_cnt/total_pixels > 0.03  # bigger than 3 percent => change detected
    orig_fps = vcap.get(cv2.CAP_PROP_FPS)
    size = tuple(map(int, (vcap.get(cv2.CAP_PROP_FRAME_WIDTH), vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    assert int(orig_fps) == fps, "Video file indicated wrong FPS"
    v = NormalVideoCapture(vcap, target_fps)
    pad = (queue_padding_in_seconds.seconds * target_fps)
    backlog = Queue(maxsize=pad)
    with tqdm(total=v.frames_left) as bar:
        last_frame = None
        while True:
            # Search for changes for a new clip
            s, f = v.read()
            if not s:
                bar.write("Finished")
                break
            if black_out_timestamps:
                f[:8, :128, :] = np.zeros((8, 128, 3))
            bar.update()
            if backlog.full():
                backlog.get()
            backlog.put(f)
            if last_frame is None:
                last_frame = f
                continue
            if not has_significant_change(f, last_frame):
                continue
            start_frame_idx = v.frames_read - 1
            timestamp_indicator = v.time_read

            # Write clip to file
            start_offset = max(timestamp_indicator - queue_padding_in_seconds, timedelta(0))
            t = date + start_offset
            bar.write(f" => T+{str(start_offset)} [frame {v.frames_read - 1}]: Found significant changes in video at {str(timestamp_indicator)} [frame {start_frame_idx}], starting from {str(t)}")
            filename = out_folder / f"{t.year}-{t.month:02}-{t.day:02}-{t.hour:02}{t.minute:02}{t.second:02}-{bed:02}-r{target_fps}.avi"
            if filename.exists():
                bar.write("  ==> Found an unfinished clip. Overwriting it.")
                filename.unlink()
            files = list(filename.parent.glob(f"{filename.stem}-*.avi"))
            if len(files) > 1:
                raise Exception(f"Found too many clips with the same timestamp: {files}")
            elif len(files) == 1:
                file = files[0]
                _, _, _, duration = parse_video_clip_file(file)
                start_timestamp = timestamp_indicator - start_offset
                duration -= start_timestamp
                duration -= queue_padding_in_seconds
                bar.write(
                    f"  ==> Found a clip with the same timestamp ({duration.total_seconds():.2f}s). Assuming it is from a previous run. Skipping {duration.total_seconds():.2f}s.")
                s, count, frames = v.skip(duration, pad)
                for f in frames:  # Refill backlog
                    if backlog.full():
                        backlog.get()
                    backlog.put(f)
                if not s:
                    bar.write("Failed to skip all frames!")
                bar.update(count)
                continue
            vw = cv2.VideoWriter(str(filename), fourcc=cv2.VideoWriter_fourcc(*'MJPG'), fps=target_fps, frameSize=size)
            backlog_buf = []
            while not backlog.empty():  # Write backlogged frames to file
                f = backlog.get()
                vw.write(f)
                backlog_buf.append(f)
            for f in backlog_buf:  # Refill backlog
                backlog.put(f)
            back_pad = 0
            end_time = v.time_read
            while back_pad < pad:
                back_pad += 1
                s, f = v.read()
                bar.update()
                if not s:
                    last_frame = f
                    bar.write("EMPTY FRAME READ!!!")
                    break
                if black_out_timestamps:
                    f[:8, :128, :] = np.zeros((8, 128, 3))
                vw.write(f)
                if backlog.full():
                    backlog.get()
                backlog.put(f)
                if has_significant_change(f, last_frame):  # We're not done yet. Extend the clip
                    back_pad = 0
                    end_time = v.time_read
                last_frame = f
            vw.release()
            frames_count = v.frames_read - start_frame_idx
            # duration2 = timedelta(seconds=frames_count / v.target_fps)
            duration = v.time_read - start_offset
            significant_frame_length = end_time - timestamp_indicator
            if significant_frame_length < MIN_SIGNIFICANT_FRAME_LENGTH:
                bar.write(
                    f"  ==> Clip ({significant_frame_length.total_seconds():.2f}s) has less than {MIN_SIGNIFICANT_FRAME_LENGTH.total_seconds():.2f}s significant frames and will be omitted")
                filename.unlink()
            else:
                new_filename = filename.with_name(f"{filename.stem}-d{int(duration.total_seconds() * 100):06}.avi")
                shutil.move(filename, new_filename)
                bar.write(
                    f"  ==> T+{str(start_offset + duration)} [frame {v.frames_read}]: Created {str(duration)} [{frames_count} frames] long clip ({str(t)} until {str(t + duration)}): {new_filename.name}")



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
        print("=== DONE ===")
        shutil.move(f, f.with_suffix(f"{f.suffix}.bak"))
        break
