"""RoomTwin scan-processing pipeline: phone LiDAR scan -> room.json (walls, openings, obstacles, QA)."""
__version__ = "0.1.0"

import os as _os

# EXR depth maps (iPhone exporters) need OpenEXR enabled before cv2 is imported
_os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
