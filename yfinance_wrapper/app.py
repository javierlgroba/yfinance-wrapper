import os
from typing import List, Optional, Type

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pyrate_limiter import Duration, Limiter, RequestRate
from requests import Session
from requests_cache import BackendSpecifier, CacheMixin, SQLiteCache
from requests_ratelimiter import (
    AbstractBucket,
    LimiterMixin,
    MemoryListBucket,
    MemoryQueueBucket,
)

from yfinance_wrapper.helpers import (
    fetch_market_data,
    process_market_data,
    tickers_data,
)
from yfinance_wrapper.parameters import HistoryParams


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


@app.get("/tickers/history/{ticker}", status_code=status.HTTP_200_OK)
def read_ticker_history(ticker: str, params: HistoryParams = Depends()) -> JSONResponse:
    market_data = fetch_market_data([ticker], params.period, params.interval, session=session)
    market_values = process_market_data(market_data)
    resp = {
        "period": params.interval,
        "interval": params.period,
        "data": {
            "ticker": ticker,
            "history": {k.tz_convert("UTC").isoformat(): v for k, v in market_values[ticker].items()},
        },
    }
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=resp,
    )


@app.get("/tickers/history", status_code=status.HTTP_200_OK)
def read_tickers_history(q: List[str] = Query(), params: HistoryParams = Depends()) -> JSONResponse:
    if len(q) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Must provide at least one ticker")
    market_data = fetch_market_data(q, params.period, params.interval, session=session)
    market_values = process_market_data(market_data)
    resp = {
        "period": params.interval,
        "interval": params.period,
        "data": [
            {
                "ticker": ticker,
                "history": {k.tz_convert("UTC").isoformat(): v for k, v in market_values[ticker].items()},
            }
            for ticker in q
        ],
    }
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=resp,
    )


@app.get("/tickers/{ticker}", status_code=status.HTTP_200_OK)
def read_ticker(ticker: str) -> JSONResponse:
    ticker_data = tickers_data([ticker], session=session)
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "data": {
                "ticker": ticker,
                **ticker_data[ticker],
            },
        },
    )


@app.get("/tickers", status_code=status.HTTP_200_OK)
def read_tickers(q: List[str] = Query()) -> JSONResponse:
    if len(q) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Must provide at least one ticker")
    ticker_data = tickers_data(q, session=session)
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "data": [
                {
                    "ticker": ticker,
                    **data,
                }
                for ticker, data in ticker_data.items()
            ],
        },
    )


def main() -> None:
    port = int(os.getenv("YFINANCE_WRAPPER_PORT", "8000"))
    host = os.getenv("YFINANCE_WRAPPER_HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)
