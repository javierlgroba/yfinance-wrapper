from typing import Any, Dict, List

import pandas as pd
import yfinance as yf
from fastapi import HTTPException, status
from requests import Session


def key(data: dict[str, Any], key: str) -> Any | None:
    return data.get(key, None)


def tickers_data(tickers: List[str], session: Session | None = None) -> dict[str, Any]:
    data = {}
    for ticker in tickers:
        ticker_data = yf.Ticker(ticker, session=session)
        info = ticker_data.info
        if "longName" not in info:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"{ticker} not found")
        fast_info = ticker_data.fast_info
        day_change = None
        if fast_info.get("currentPrice", None) is not None and fast_info.get("previousClose", None) is not None:
            day_change = fast_info["currentPrice"] - fast_info["previousClose"]

        data[ticker] = {
            "name": key(info, "longName"),
            "price": key(fast_info, "currentPrice"),
            "currency": key(fast_info, "currency"),
            "previous_close": key(info, "previousClose"),
            "open": key(fast_info, "open"),
            "day_high": key(fast_info, "dayHigh"),
            "day_low": key(fast_info, "dayLow"),
            "day_change": day_change,
            "year_low": key(fast_info, "yearLow"),
            "year_high": key(fast_info, "yearHigh"),
        }
        print(data)
    return data


def fetch_market_data(tickers: List[str], period: str, interval: str, session: Session | None = None) -> pd.DataFrame:
    hist = yf.download(tickers, period=period, interval=interval, session=session)
    if hist.empty:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Cannot fetch data for {tickers}")
    return hist["Adj Close"]


def process_market_data(market_data: pd.DataFrame) -> Dict[str, Any]:
    market_values: Dict[str, Any] = {}
    for date, row in market_data.iterrows():
        for ticker, price in row.items():
            if pd.isna(price) or pd.isnull(price):
                continue
            if ticker not in market_values:
                market_values[ticker] = {}
            market_values[ticker][date] = price
    if len(market_values) == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No data found")
    return market_values
