import os
from datetime import datetime, timezone
from typing import Any, Optional, Type

import pandas as pd
import uvicorn
import yfinance as yf
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator, model_validator
from pyrate_limiter import Duration, Limiter, RequestRate
from requests import Session
from requests_cache import BackendSpecifier, CacheMixin, SQLiteCache
from requests_ratelimiter import (
    AbstractBucket,
    LimiterMixin,
    MemoryListBucket,
    MemoryQueueBucket,
)


class CachedLimiterSession(CacheMixin, LimiterMixin, Session):
    """
    Session class with caching and rate-limiting behavior. Accepts arguments for both
    LimiterSession and CachedSession.
    """

    def __init__(
        self,
        limiter: Optional[Limiter] = None,
        backend: Optional[BackendSpecifier] = None,
        bucket_class: Type[AbstractBucket] = MemoryListBucket,
    ):
        CacheMixin.__init__(self, backend=backend, limiter=limiter)
        LimiterMixin.__init__(self, limiter=limiter, bucket_class=bucket_class)


session = CachedLimiterSession(
    limiter=Limiter(RequestRate(2, Duration.SECOND * 5)),
    bucket_class=MemoryQueueBucket,
    backend=SQLiteCache("yfinance.cache"),
)


def key(data: dict[str, Any], key: str) -> Any | None:
    return data.get(key, None)


app = FastAPI()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": "Invalid request",
            "errors": exc.errors(),  # Use the default errors for more detail
            "request_body": exc.body,  # Include the request body that caused the error
        },
    )


@app.get("/", status_code=status.HTTP_200_OK)
def read_root(response: Response) -> Response:
    response.status_code = status.HTTP_200_OK
    return response


@app.get("/ready", status_code=status.HTTP_200_OK)
def read_ready(response: Response) -> Response:
    response.status_code = status.HTTP_200_OK
    return response


@app.get("/live", status_code=status.HTTP_200_OK)
def read_live(response: Response) -> Response:
    response.status_code = status.HTTP_200_OK
    return response


@app.get("/ticker/{ticker}", status_code=status.HTTP_200_OK)
def read_ticker(ticker: str) -> JSONResponse:
    ticker_data = yf.Ticker(ticker, session=session)
    info = ticker_data.info
    if "longName" not in info:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"{ticker} not found")
    fast_info = ticker_data.fast_info
    day_change = None
    if "lastPrice" in fast_info and "previousClose" in fast_info:
        day_change = fast_info["lastPrice"] - fast_info["previousClose"]

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "ticker": ticker,
            "name": key(info, "longName"),
            "price": key(fast_info, "lastPrice"),
            "currency": key(fast_info, "currency"),
            "previous_close": key(info, "previousClose"),
            "open": key(fast_info, "open"),
            "day_high": key(fast_info, "dayHigh"),
            "day_low": key(fast_info, "dayLow"),
            "day_change": day_change,
            "year_low": key(fast_info, "yearLow"),
            "year_high": key(fast_info, "yearHigh"),
        },
    )


class HistoryParams(BaseModel):
    interval: str = "1mo"
    period: str = "max"

    @field_validator("interval")
    @classmethod
    def validate_interval(cls: Type["HistoryParams"], interval: str) -> str:
        valid_intervals = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]
        if interval not in valid_intervals:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid interval: {interval}")
        return interval

    @field_validator("period")
    @classmethod
    def validate_period(cls: Type["HistoryParams"], period: str) -> str:
        valid_periods = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
        if period not in valid_periods:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid period: {period}")
        return period

    @model_validator(mode="after")
    def validate_period_interval(self: "HistoryParams") -> "HistoryParams":
        # From: https://github.com/ranaroussi/yfinance/wiki/Ticker#parameters
        if self.interval in ["1m", "2m", "5m", "15m", "30m"] and self.period not in ["1d", "5d"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="For intervals smaller than 1 hour, period must be 1 week or smaller",
            )
        elif self.interval in ["60m", "90m", "1h"] and self.period not in ["1d", "5d", "1mo"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="For intervals smaller than 1 day, period must be 1 month or smaller",
            )
        return self


@app.get("/ticker/{ticker}/history", status_code=status.HTTP_200_OK)
def read_ticker_history(ticker: str, params: HistoryParams = Depends()) -> JSONResponse:
    hist = yf.download([ticker], period=params.period, interval=params.interval, session=session)
    if hist.empty:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No data found for {ticker}")
    resp = {
        "ticker": ticker,
        "period": params.interval,
        "interval": params.period,
        "data": {k.tz_convert("UTC").isoformat(): v for k, v in hist["Adj Close"][ticker].to_dict().items()},
    }
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=resp,
    )


class Trade(BaseModel):
    date_time: datetime
    quantity: float


class Portfolio(BaseModel):
    trades: dict[str, list[Trade]]

    @field_validator("trades")
    @classmethod
    def validate_trades(cls: Type["Portfolio"], trades: dict[str, list[Trade]]) -> dict[str, list[Trade]]:
        if len(trades) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Portfolio must have at least one trade"
            )
        keys = []
        for k, v in trades.items():
            if k in keys:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Duplicate trade key: {k}")
            keys.append(k)
            if len(v) == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Trade list for {k} must have at least one trade",
                )
            for trade in v:
                if trade.quantity <= 0:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST, detail=f"Trade amount must be positive: {trade.amount}"
                    )
                if trade.date_time.tzinfo is None:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Trade date must have timezone: {trade.date_time}",
                    )
                if trade.date_time > datetime.now(tz=timezone.utc):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Trade date must be in the past: {trade.date_time}",
                    )
            v.sort(key=lambda x: x.date_time)
        return trades


@app.post("/portfolio/history", status_code=status.HTTP_201_CREATED)
def read_portfolio_history(request: Portfolio, params: HistoryParams = Depends()) -> JSONResponse:
    total_owned = {}
    tickers = [k for k in request.trades.keys()]
    hist = yf.download(tickers, period=params.period, interval=params.interval, session=session)
    if hist.empty:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Cannot fetch data for {tickers}")
    market_values = hist["Adj Close"].to_dict()
    for ticker, values in market_values.items():
        close_prices = list(values.items())
        if pd.isna(close_prices[-1][1]):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"No data found for {ticker}")
        trades = request.trades[ticker]
        trade_idx = 0
        quantity = 0.0
        for i in range(len(close_prices)):
            before = close_prices[i][0]
            after = close_prices[i + 1][0] if i + 1 < len(close_prices) else datetime.now(tz=timezone.utc)
            if (
                trade_idx < len(trades)
                and trades[trade_idx].date_time >= before
                and trades[trade_idx].date_time < after
            ):
                quantity = trades[trade_idx].quantity
                trade_idx += 1
            if before not in total_owned:
                total_owned[before] = 0
            close_price = close_prices[i][1]
            total_owned[before] += (
                (quantity * close_price) if not pd.isna(close_price) or not pd.isnull(close_price) else 0
            )
    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content={k.tz_convert("UTC").isoformat(): v for k, v in total_owned.items()},
    )


def main() -> None:
    port = int(os.getenv("YFINANCE_WRAPPER_PORT", "8000"))
    host = os.getenv("YFINANCE_WRAPPER_HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)
