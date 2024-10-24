###########
# FLASK #
###########

from flask import Flask, render_template, request, jsonify

###########
# PLOTTING #
###########

import plotly.graph_objs as go
import pandas as pd

##########
# FINANCE #
##########

import yfinance as yf
import requests

###########
# UTILITIES #
###########

import os
import sys
import datetime
import math
import logging
import datetime
from dotenv import load_dotenv

##########
# OPENAI #
##########

import openai
import torch
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

##########
# ALPACA #
##########

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
