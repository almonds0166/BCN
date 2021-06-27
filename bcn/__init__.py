
from collections import namedtuple

from .bcn import *

VersionInfo = namedtuple("VersionInfo", "major minor build")
version_info = VersionInfo(
   major=0,
   minor=0,
   build=1
)

__title__ = "BCN"
__author__ = "Madison Landry"
__copyright__ = "Copyright 2021-present Madison Landry"
__version__ = f"{version_info.major}.{version_info.minor}.{version_info.build}"
