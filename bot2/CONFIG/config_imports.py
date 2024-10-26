###########
# FLASK #
###########

from flask import Flask, render_template, request, jsonify

###########
# PLOTTING #
###########

import plotly.graph_objs as go
import pandas as pd
import matplotlib.pyplot as plt

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
from datetime import datetime, timedelta
import time
import math
import logging
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


import yfinance as yf
import pandas as pd
import numpy as np
from hmmlearn import hmm
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import openai  # ChatGPT API (openai-python library)
import os

################
# PROGRESS BAR #
################

from tqdm import tqdm 

################
# ML AND STATS #
################

import statsmodels.api as sm
from hmmlearn import hmm
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from scipy.optimize import minimize