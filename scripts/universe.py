"""Ticker lists used before the full universe is built.

The Yahoo symbols below are candidates. The source check reports any symbol
that returns no data, so a wrong or renamed ticker is found early.
"""

# Expected size of the final universe: S&P 500 members plus European peers.
FULL_UNIVERSE_SIZE = 550

BENCHMARKS = ["^GSPC", "SPY", "^STOXX", "IGV"]

# US groups named in the AI-scare events we start from.
AI_SENSITIVE_US = [
    # Financial data and ratings
    "FDS", "SPGI", "MCO", "MSCI",
    # Application software
    "CRM", "ADBE", "NOW", "INTU",
    # Insurance brokers
    "AON", "AJG", "WTW", "BRO",
    # Wealth managers
    "SCHW", "RJF", "LPLA",
    # Property services
    "CBRE", "JLL",
    # Freight logistics
    "CHRW", "EXPD", "JBHT", "ODFL",
    # Gaming
    "TTWO", "EA",
]

EUROPEAN_PEERS = [
    # Information services
    "REL.L", "WKL.AS", "TRI.TO", "LSEG.L", "EXPN.L", "INF.L", "PSON.L",
    # Software and IT services
    "SAP.DE", "SGE.L", "DSY.PA", "CAP.PA", "TEP.PA",
    # Advertising
    "PUB.PA", "WPP.L",
    # Staffing and wealth management
    "RAND.AS", "SJP.L",
    # Freight logistics
    "DSV.CO", "KNIN.SW", "DHL.DE",
    # Gaming
    "UBI.PA",
]

# Used for the Wikimedia check: English Wikipedia article titles.
SAMPLE_ARTICLES = ["RELX", "Wolters_Kluwer", "FactSet"]

# Used for the EDGAR and earnings checks.
SAMPLE_US_FILERS = ["FDS", "SPGI", "CHRW"]
SAMPLE_EARNINGS_TICKERS = ["FDS", "CHRW", "WKL.AS"]

FRED_SERIES = ["DGS10", "DGS2", "VIXCLS"]
