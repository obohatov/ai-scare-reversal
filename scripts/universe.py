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
    "RAND.AS", "STJ.L",
    # Freight logistics
    "DSV.CO", "KNIN.SW", "DHL.DE",
    # Gaming
    "UBI.PA",
]

# English Wikipedia titles for peers; redirects are resolved at download time.
EUROPEAN_ARTICLES = {
    "REL.L": "RELX",
    "WKL.AS": "Wolters_Kluwer",
    "TRI.TO": "Thomson_Reuters",
    "LSEG.L": "London_Stock_Exchange_Group",
    "EXPN.L": "Experian",
    "INF.L": "Informa",
    "PSON.L": "Pearson_plc",
    "SAP.DE": "SAP",
    "SGE.L": "Sage_Group",
    "DSY.PA": "Dassault_Syst\u00e8mes",
    "CAP.PA": "Capgemini",
    "TEP.PA": "Teleperformance",
    "PUB.PA": "Publicis",
    "WPP.L": "WPP_plc",
    "RAND.AS": "Randstad_NV",
    "STJ.L": "St._James's_Place_plc",
    "DSV.CO": "DSV_A/S",
    "KNIN.SW": "Kuehne_+_Nagel",
    "DHL.DE": "DHL_Group",
    "UBI.PA": "Ubisoft",
}

# Index members whose symbol changed: old symbol -> symbol that Yahoo serves now.
TICKER_RENAMES = {
    "ABC": "COR",
    "ADS": "BFH",
    "ANTM": "ELV",
    "BHGE": "BKR",
    "BK": "BNY",
    "BLL": "BALL",
    "FB": "META",
    "FI": "FISV",
    "FLT": "CPAY",
    "GPS": "GAP",
    "HFC": "DINO",
    "MMC": "MRSH",
    "NLOK": "GEN",
    "PEAK": "DOC",
    "PKI": "RVTY",
    "RE": "EG",
    "SYMC": "GEN",
    "TMK": "GL",
    "UTX": "RTX",
    "VIAC": "PARA",
    "WLTW": "WTW",
}

# Stocks named in AI-scare events that may sit outside the S&P 500.
AI_EVENT_EXTRAS = [
    "CHGG", "HUBS", "JLL", "LPLA", "NTSK", "QLYS", "RBLX", "RPD", "TENB", "XPO", "ZS",
]

# Used for the Wikimedia check: English Wikipedia article titles.
SAMPLE_ARTICLES = ["RELX", "Wolters_Kluwer", "FactSet"]

# Used for the EDGAR and earnings checks.
SAMPLE_US_FILERS = ["FDS", "SPGI", "CHRW"]
SAMPLE_EARNINGS_TICKERS = ["FDS", "CHRW", "WKL.AS"]

FRED_SERIES = ["DGS10", "DGS2", "VIXCLS", "DFF"]
