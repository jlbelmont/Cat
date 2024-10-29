from bot2.CONFIG.config_imports import * 

# Load environment variables
load_dotenv()

######################
# API INITIALIZATION #
######################

# Global variables for storing the API clients
trading_client = None
data_client = None
assets_request = None
api = None
stream = None  # Added `stream` as a global variable

# ALPACA API Keys and URL
ALP_API_KEY = os.getenv('ALPACA_API_KEY')
ALP_API_SECRET = os.getenv('ALPACA_SECRET_KEY')
ALP_BASE_URL = 'https://paper-api.alpaca.markets'  # For paper trading

# Debugging API Key loading
if not ALP_API_KEY or not ALP_API_SECRET:
    print("Error: API keys are not loaded correctly.")
else:
    print("API keys loaded successfully.")

def initialize_clients():
    """
    Initialize the Alpaca API clients only once and store them in global variables.
    This ensures the API clients are only initialized once, even if this file is imported multiple times.
    """
    global trading_client, data_client, assets_request, api, stream  # Added `stream` and `api` as globals
    if trading_client is None:
        print("Initializing Alpaca API clients...")
        trading_client = TradingClient(ALP_API_KEY, ALP_API_SECRET, paper=True)
        data_client = StockHistoricalDataClient(ALP_API_KEY, ALP_API_SECRET)
        assets_request = GetAssetsRequest(asset_class=AssetClass.US_EQUITY)
        api = tradeapi.REST(
            key_id=ALP_API_KEY,
            secret_key=ALP_API_SECRET,
            base_url=ALP_BASE_URL,
            api_version='v2'
        )
        # Initialize the streaming client
        stream = TradingStream(ALP_API_KEY, ALP_API_SECRET, paper=True)
        print("Alpaca API clients successfully initialized.")
    else:
        print("Alpaca API clients already initialized.")

# Initialize the clients when this module is imported
initialize_clients()


#################
# LLMS API KEYS #
#################

HF_KEY = os.getenv('HF_KEY')
OAI_KEY = os.getenv('OAI_KEY')

