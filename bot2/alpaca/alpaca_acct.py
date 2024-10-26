#########
# GOALS #
#########

#> get account information, feed to display

###########
# IMPORTS #
###########

from alpaca_realtime import * 

################
# ACCOUNT INFO #
################

def get_account_information():
    """
    Fetches account information from Alpaca's API and returns a dictionary of relevant details.
    
    Returns:
        dict: A dictionary containing account information (e.g., cash balance, buying power, etc.).
    """
    try:
        # Fetch the account information using Alpaca's trading client
        account = trading_client.get_account()
        
        # Extract relevant account details to display on the front-end
        account_info = {
            'cash': account.cash,
            'buying_power': account.buying_power,
            'portfolio_value': account.portfolio_value,
            'equity': account.equity,
            'last_equity': account.last_equity,
            'long_market_value': account.long_market_value,
            'short_market_value': account.short_market_value,
            'initial_margin': account.initial_margin,
            'maintenance_margin': account.maintenance_margin,
            'status': account.status
        }
        
        return account_info
    
    except Exception as e:
        print(f"Error fetching account information: {e}")
        return {}

#########################
# FRONTEND DATA PASSING #
#########################

def pass_account_info_to_frontend():
    """
    Prepares the account information to be sent to the frontend for display.
    
    Returns:
        dict: Account information formatted for the frontend.
    """
    account_info = get_account_information()
    
    # Additional processing can be done here before passing to the frontend, if needed.
    
    return account_info

################################
# GAINS AND LOSSES CALCULATION #
################################

def calculate_gains_or_losses():
    """
    Calculate the daily and overall gains or losses based on the account's equity and portfolio value.
    
    Returns:
        dict: A dictionary containing the daily and overall gains/losses.
    """
    try:
        # Fetch account information
        account = trading_client.get_account()

        # Calculate daily gains/losses
        daily_gain_loss = float(account.equity) - float(account.last_equity)
        
        # Calculate overall gains/losses
        overall_gain_loss = float(account.equity) - float(account.portfolio_value)
        
        # Prepare the result dictionary
        gain_loss_info = {
            'daily_gain_loss': daily_gain_loss,
            'overall_gain_loss': overall_gain_loss
        }
        
        return gain_loss_info
    
    except Exception as e:
        print(f"Error calculating gains/losses: {e}")
        return {}

###################
# ROI CALCULATION #
###################

def calculate_roi():
    """
    Calculate the return on investment (ROI) for the account.
    
    Returns:
        float: The ROI as a percentage.
    """
    try:
        # Fetch account information
        account = trading_client.get_account()
        
        # Calculate ROI based on current equity and the starting portfolio value
        initial_value = float(account.portfolio_value)
        current_equity = float(account.equity)
        roi = ((current_equity - initial_value) / initial_value) * 100
        
        return roi
    
    except Exception as e:
        print(f"Error calculating ROI: {e}")
        return 0.0

########################################
# BUYING POWER UTILIZATION CALCULATION #
########################################

def calculate_buying_power_utilization():
    """
    Calculate the percentage of the account's buying power being utilized.
    
    Returns:
        float: The percentage of buying power utilization.
    """
    try:
        # Fetch account information
        account = trading_client.get_account()
        
        # Calculate how much buying power is being used
        total_buying_power = float(account.buying_power)
        equity = float(account.equity)
        
        buying_power_utilization = ((equity - total_buying_power) / equity) * 100
        
        return buying_power_utilization
    
    except Exception as e:
        print(f"Error calculating buying power utilization: {e}")
        return 0.0

#######################################
# LONG AND SHORT MARKET EXPOSURE INFO #
#######################################

def get_market_exposure():
    """
    Retrieve long and short market exposure for the account.
    
    Returns:
        dict: A dictionary containing long and short market exposure values.
    """
    try:
        # Fetch account information
        account = trading_client.get_account()
        
        # Get long and short market values
        long_market_value = float(account.long_market_value)
        short_market_value = float(account.short_market_value)
        
        exposure_info = {
            'long_market_value': long_market_value,
            'short_market_value': short_market_value
        }
        
        return exposure_info
    
    except Exception as e:
        print(f"Error fetching market exposure: {e}")
        return {}

##############################
# PORTFOLIO COMPOSITION INFO #
##############################

def get_portfolio_composition():
    """
    Fetch the current holdings in the portfolio.
    
    Returns:
        list: A list of dictionaries, each containing details of a stock holding.
    """
    try:
        # Get the current positions in the account
        positions = trading_client.get_all_positions()
        
        # Extract relevant information for each position
        holdings = []
        for position in positions:
            holdings.append({
                'symbol': position.symbol,
                'quantity': position.qty,
                'market_value': position.market_value,
                'avg_entry_price': position.avg_entry_price,
                'current_price': position.current_price,
                'unrealized_pl': position.unrealized_pl
            })
        
        return holdings
    
    except Exception as e:
        print(f"Error fetching portfolio composition: {e}")
        return []

##########################################
# GATHER ALL ACCOUNT METRICS FOR DISPLAY #
##########################################

def get_full_account_display_data():
    """
    Gathers all relevant account metrics for display in the frontend.
    
    Returns:
        dict: A dictionary containing all the key metrics (account info, gains/losses, ROI, etc.).
    """
    account_info = get_account_information()
    gain_loss_info = calculate_gains_or_losses()
    roi = calculate_roi()
    buying_power_utilization = calculate_buying_power_utilization()
    market_exposure = get_market_exposure()
    portfolio_composition = get_portfolio_composition()

    # Combine all the data into a single dictionary
    display_data = {
        'account_info': account_info,
        'gains_losses': gain_loss_info,
        'roi': roi,
        'buying_power_utilization': buying_power_utilization,
        'market_exposure': market_exposure,
        'portfolio_composition': portfolio_composition
    }
    
    return display_data
