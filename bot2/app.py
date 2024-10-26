###########
# IMPORTS #
###########

from bot2.CONFIG import (
    config_vars
)

from bot2.CONFIG import config_imports
import alpaca.alpaca_acct, data.scraper, signals_n_strats.hist_anal, hmm.hmm, llms.gpt_llm, llms.hf_llm, nlp.nlp, vis.vis

##########
# ROUTES #
##########

@app.route('/', methods = ['GET', 'POST'])
def home():
    
    ######################
    # OVERVIEW GOES HERE #
    ######################
    
    return render_template('home.html')

@app.route('/alpaca', methods = ['GET', 'POST'])
def alpaca():
    
    ####################
    # LLMS GO HERE TOO #
    ####################
    
    #######################
    # STATS & VIS GO HERE #
    #######################
    
    return render_template('alpaca.html')

@app.route('/proofs', methods = ['GET', 'POST'])
def proofs():
    
    ###########################
    # STATIC PAGE MOST LIKELY #
    ###########################
    
    return render_template('proofs.html')

@app.route('/stat_hell', methods = ['GET', 'POST'])
def stat_hell():
    
    ###################
    # A LOT GOES HERE #
    ###################
    
    return render_template('stat_hell.html')