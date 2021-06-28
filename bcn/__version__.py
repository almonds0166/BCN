from collections import namedtuple

VersionInfo = namedtuple("VersionInfo", "major minor build")
version_info = VersionInfo(
   major=0,
   minor=2,
   build=18,
)
__version__ = f"{version_info.major}.{version_info.minor}.{version_info.build}"