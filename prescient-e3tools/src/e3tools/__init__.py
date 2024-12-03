from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("prescient-e3tools")
except PackageNotFoundError:
    __version__ = "unknown version"
