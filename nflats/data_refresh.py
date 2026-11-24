"""python -m nflats.data_refresh 2026 : refresh nflverse data for seasons 2009..N."""
import sys
from . import data
if __name__ == "__main__":
    last = int(sys.argv[1])
    data.refresh(list(range(2009, last + 1)))
    print("refreshed")
