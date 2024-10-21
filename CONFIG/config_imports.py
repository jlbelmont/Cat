from flask import Flask, render_template, request, jsonify
import plotly.graph_objs as go
import pandas as pd
import os
import requests
import yfinance as yf
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass, OrderSide, TimeInForce, OrderStatus, OrderClass
from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
    GetOrdersRequest,
    TrailingStopOrderRequest,
    TakeProfitRequest,
    StopLossRequest
)
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import datetime
import math
import logging
import openai
import torch
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer