from typing import Type

from fastapi import HTTPException, status
from pydantic import BaseModel, field_validator, model_validator


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
