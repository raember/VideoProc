import math

import cv2
import numpy as np
from datetime import timedelta
from typing import Tuple, Optional

v = cv2.VideoCapture()

class NormalVideoCapture(cv2.VideoCapture):
    cap: cv2.VideoCapture
    def __init__(self, cap: cv2.VideoCapture, target_fps: float):
        super(NormalVideoCapture, self).__init__()
        self.cap = cap
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.target_fps = target_fps
        self.total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self._cur_pos = 0.0
        self._cur_frame = None
        self._next_pos = 0.0
        self._next_frame = None
        self._read_count = 0
        self._orig_frames_read = 0

    @property
    def incr(self) -> float:
        return self.fps/self.target_fps
    
    @property
    def pos(self) -> float:
        return self._next_pos

    @property
    def frames_read(self) -> int:
        return self._read_count

    @property
    def frames_left(self) -> int:
        return math.ceil((self.frames_left_orig + 1) / self.incr)

    @property
    def frames_left_orig(self) -> int:
        return self.total - self._next_pos

    @property
    def time_left(self) -> timedelta:
        return timedelta(seconds=self.frames_left_orig / self.fps)

    @property
    def time_read(self) -> timedelta:
        return timedelta(seconds=self.pos / self.fps)

    def read_original(self, use_fallback: bool = False) -> Optional[np.ndarray]:
        s, f = self.cap.read()
        # print(f"==> Read {nth(self._orig_frames_read)} frame")
        if not s:
            if use_fallback:
                return self._cur_frame
            return None
        self._orig_frames_read += 1
        return f

    def _return_frame(self, f: Optional[np.ndarray]) -> Tuple[bool, Optional[np.ndarray]]:
        if f is not None:
            self._read_count += 1
        return f is not None, f

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        # print("#" * 100, self._read_count, "#" * 100)
        # If at the beginning, read the first frame so the rest of the code doesn't go up in flames
        if self._cur_frame is None:
            # print("=> First frame")
            self._cur_frame = self.read_original()
            return self._return_frame(self._cur_frame)  # First frame is always the same as the source

        # Update position
        self._cur_pos, self._cur_frame = self._next_pos, self._next_frame
        self._next_pos += self.incr
        # print(f"POS: {self._cur_pos}")
        # print(f"NEXT POS: {self._next_pos}")

        # Skip unneeded frames
        skip_n = math.ceil(round(self._next_pos, 2)) - math.ceil(round(self._cur_pos, 2))
        # print(f"=> Skipping {skip_n} frames")
        for _ in range(skip_n):
            if self._next_frame is not None:
                self._cur_frame = self._next_frame
                self._next_frame = None
            else:
                self._cur_frame = self.read_original()
                if self._cur_frame is None:
                    return False, None

        weight = round(self._next_pos % 1.0, 2)  # weight = intra-frame position
        # print(f"Weight: {weight}")
        # If the intra-frame position is 0.0, the position coincides with a frame of the original
        # => no need to interpolate
        if weight == 0.0:
            # print("=> Returning original frame")
            return self._return_frame(self._cur_frame)

        # Read the next frame to interpolate with
        # print("=> Reading next frame to interpolate")
        if (f := self.read_original(skip_n > 0)) is None:
            return False, None
        self._next_frame = f

        # Interpolate between the previous and the new frame according to the weight
        # print("=> Interpolating frames")
        new_f = cv2.addWeighted(self._cur_frame, weight, self._next_frame, 1.0 - weight, 0.0)
        return self._return_frame(new_f)

    def skip(self, duration: timedelta) -> Tuple[bool, int]:
        success = True
        count = 0
        for _ in range(int(duration.total_seconds() * self.fps)):
            success &= self.cap.grab()
            count += 1
            self._orig_frames_read += 1
        # I hope this is accurate enough
        offset = duration.total_seconds() * self.target_fps
        self._cur_pos += offset
        self._next_pos += offset
        return success, count

def nth(n: int) -> str:
    s = str(n)
    if s.endswith('1'):
        return f"{n}st"
    if s.endswith('2'):
        return f"{n}nd"
    if s.endswith('3'):
        return f"{n}rd"
    return f"{n}th"
