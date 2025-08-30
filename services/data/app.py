
from ohlcv import load_ohlcv
from sentiment import fetch_cp_headlines, score_headlines
btc = load_ohlcv("BTC", 540)
cp = fetch_cp_headlines(auth_token="9d497b2b6e9cf367ca46234ba74feac3a7d7ea73")
sent = score_headlines(cp)  

def main():
    print("BTC OHLCV sample:")
    print(btc.head())
    print("\nCryptopanic headlines sample:")
    print(cp.head()) 
    print("\nSentiment time series (ETH, if present):")
    if "ETH" in sent:
        print(sent["ETH"].head())
    else:
        print("No ETH sentiment data.")

if __name__ == "__main__":
    main()