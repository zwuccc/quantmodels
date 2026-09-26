"""MarketData: everything a strategy is allowed to see, clippable at any date."""
from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd

from qm.data.fundamentals import Fundamentals


@dataclass
class MarketData:
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    fundamentals: Fundamentals
    universe: pd.DataFrame  # ticker, sector, kind

    def clip(self, end) -> "MarketData":
        """Only data known by the close of `end`: prices up to end, fundamentals filed by end."""
        end = pd.Timestamp(end)
        cut = {k: getattr(self, k).loc[:end] for k in ("open", "high", "low", "close", "volume")}
        return replace(self, **cut, fundamentals=self.fundamentals.clip(end))

    @property
    def calendar(self) -> pd.DatetimeIndex:
        return self.close.index

    def tickers(self, kind: str) -> list[str]:
        u = self.universe
        names = u.loc[u["kind"] == ("stock" if kind == "stocks" else kind), "ticker"]
        return [t for t in names if t in self.close.columns]

    def sector(self) -> pd.Series:
        return self.universe.set_index("ticker")["sector"]
