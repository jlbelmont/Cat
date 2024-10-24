from bot2.CONFIG.config_imports import * 

load_dotenv()

#############
# API STUFF #
##############

# ALPACA #

ALP_API_KEY = os.getenv('ALPACA_API_KEY')
ALP_API_SECRET = os.getenv('ALPACA_SECRET_KEY')
ALP_BASE_URL = 'https://paper-api.alpaca.markets'  # For paper trading
trading_client = TradingClient(ALP_API_KEY, ALP_API_SECRET, paper=True)
data_client = StockHistoricalDataClient(ALP_API_KEY, ALP_API_SECRET)
assets_request = GetAssetsRequest(asset_class=AssetClass.US_EQUITY)

# LLMS #

HF_KEY = os.getenv('HF_KEY')
OAI_KEY = os.getenv('OAI_KEY')


