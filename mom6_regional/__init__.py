from pkg_resources import get_distribution, DistributionNotFound

try:
    __version__ = get_distribution("mom6-regional").version
except DistributionNotFound:
    pass
