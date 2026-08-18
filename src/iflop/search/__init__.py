"""Search shell for I-FLOP algorithms."""

from iflop.search.flop_search import run_flop_search
from iflop.search.iflop import run_iflop_search
from iflop.search.state import SearchResult

__all__ = [
    "SearchResult",
    "run_flop_search",
    "run_iflop_search",
]
