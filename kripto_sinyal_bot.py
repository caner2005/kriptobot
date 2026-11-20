#!/usr/bin/env python3
"""
Kripto AL/SAT sinyal botu (sadece sinyal üretir, emir GÖNDERMEZ).

Strateji (4h/1h gibi kapanmış mumlar üzerinde):
  Trend   : EMA20 > EMA50 (yukarı) / EMA20 < EMA50 (aşağı)
  Momentum: MACD sinyal çizgisini keser
  Aşırı   : RSI14 (AL için <70, SAT için >30)
  Skor    : her koşul 1 puan, >= MIN_SKOR ise sinyal
  Risk    : ATR(14) tabanlı stop ve hedef önerisi

Kurulum:   pip install ccxt pandas numpy requests
Canlı:     python kripto_sinyal_bot.py --semboller BTC/USDT ETH/USDT --zaman 1h
Backtest:  python kripto_sinyal_bot.py --backtest --semboller BTC/USDT --zaman 4h
Demo:      python kripto_sinyal_bot.py --demo     (internetsiz, sahte veri)
Telegram:  TG_TOKEN ve TG_CHAT_ID ortam değişkenlerini ayarlarsan uyarı gönderir.

Yasal not: Yatırım tavsiyesi değildir. Geçmiş performans geleceği garanti etmez.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

MIN_SKOR = 3          # 0-4 arası; yüksek = daha az ama daha seçici sinyal
ATR_STOP = 1.5        # stop = giriş -/+ 1.5 x ATR
ATR_HEDEF = 3.0       # hedef = giriş +/- 3 x ATR  (R:R = 1:2)


# ---------------------------------------------------------------- indikatörler
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def atr(df, n=14):
    pc = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - pc).abs(),
                    (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def hesapla(df):
    df = df.copy()
    df["ema_h"] = ema(df["close"], 20)
    df["ema_y"] = ema(df["close"], 50)
    macd = ema(df["close"], 12) - ema(df["close"], 26)
    df["macd"] = macd
    df["macd_sig"] = ema(macd, 9)
    df["rsi"] = rsi(df["close"])
    df["atr"] = atr(df)

    macd_yukari = (df["macd"] > df["macd_sig"]) & (df["macd"].shift() <= df["macd_sig"].shift())
    macd_asagi = (df["macd"] < df["macd_sig"]) & (df["macd"].shift() >= df["macd_sig"].shift())
    macd_poz = df["macd"] > df["macd_sig"]

    # AL skoru: trend + macd kesişim (yeni) + macd pozitif bölge + rsi uygun
    df["al_skor"] = ((df["ema_h"] > df["ema_y"]).astype(int)
                     + macd_yukari.astype(int) * 2          # kesişim tetikleyici, 2 puan
                     + (df["rsi"] < 70).astype(int))
    df["sat_skor"] = ((df["ema_h"] < df["ema_y"]).astype(int)
                      + macd_asagi.astype(int) * 2
                      + (df["rsi"] > 30).astype(int))
    # tetikleyici (kesişim) olmadan sinyal yok
    df["sinyal"] = 0
    # RSI zorunlu filtre: aşırı alımda AL, aşırı satımda SAT verme
    df.loc[macd_yukari & (df["al_skor"] >= MIN_SKOR) & (df["rsi"] < 70), "sinyal"] = 1
    df.loc[macd_asagi & (df["sat_skor"] >= MIN_SKOR) & (df["rsi"] > 30), "sinyal"] = -1
    return df


# ---------------------------------------------------------------------- veri
def veri_cek(borsa, sembol, zaman, limit=300):
    o = borsa.fetch_ohlcv(sembol, timeframe=zaman, limit=limit)
    df = pd.DataFrame(o, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    return df.set_index("ts")


def sahte_veri(n=600, seed=7):
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0003, 0.012, n) + 0.01 * np.sin(np.arange(n) / 40)
    c = 30000 * np.exp(np.cumsum(r))
    h = c * (1 + rng.uniform(0, 0.006, n))
    l = c * (1 - rng.uniform(0, 0.006, n))
    o = np.r_[c[0], c[:-1]]
    idx = pd.date_range("2026-01-01", periods=n, freq="4h")
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": 1.0}, index=idx)


# ------------------------------------------------------------------ bildirim
def bildir(mesaj):
    print(mesaj, flush=True)
    tok, chat = os.getenv("TG_TOKEN"), os.getenv("TG_CHAT_ID")
    if tok and chat:
        try:
            import requests
            requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                          data={"chat_id": chat, "text": mesaj}, timeout=10)
        except Exception as e:
            print("Telegram hatası:", e)


def mesaj_olustur(sembol, zaman, satir):
    yon = "🟢 AL" if satir["sinyal"] == 1 else "🔴 SAT"
    p, a = satir["close"], satir["atr"]
    if satir["sinyal"] == 1:
        stop, hedef = p - ATR_STOP * a, p + ATR_HEDEF * a
    else:
        stop, hedef = p + ATR_STOP * a, p - ATR_HEDEF * a
    return (f"{yon} {sembol} [{zaman}]\nFiyat: {p:.4f}\nRSI: {satir['rsi']:.1f}\n"
            f"Stop: {stop:.4f} | Hedef: {hedef:.4f}\n(Yatırım tavsiyesi değildir)")


# ------------------------------------------------------------------ backtest
def backtest(df, komisyon=0.001):
    """Basit: AL sinyalinde long gir, ATR stop/hedef ya da ters sinyalde çık. Sadece long."""
    d = hesapla(df)
    poz, islemler = None, []
    for i in range(60, len(d) - 1):
        s = d.iloc[i]
        nxt = d.iloc[i + 1]
        if poz:
            # önce stop/hedef (aynı mumda ikisi de olursa kötümser: stop)
            if nxt["low"] <= poz["stop"]:
                islemler.append(poz["stop"] / poz["giris"] - 1 - 2 * komisyon); poz = None
            elif nxt["high"] >= poz["hedef"]:
                islemler.append(poz["hedef"] / poz["giris"] - 1 - 2 * komisyon); poz = None
            elif s["sinyal"] == -1:
                islemler.append(nxt["open"] / poz["giris"] - 1 - 2 * komisyon); poz = None
        elif s["sinyal"] == 1:
            g = nxt["open"]
            poz = {"giris": g, "stop": g - ATR_STOP * s["atr"], "hedef": g + ATR_HEDEF * s["atr"]}
    if not islemler:
        return "İşlem yok."
    x = np.array(islemler)
    ekin = np.cumprod(1 + x)
    dd = (ekin / np.maximum.accumulate(ekin) - 1).min()
    return (f"İşlem: {len(x)} | Kazanma oranı: {(x > 0).mean():.0%} | "
            f"Toplam getiri: {ekin[-1] - 1:+.1%} | Max düşüş: {dd:.1%} | "
            f"Ort. işlem: {x.mean():+.2%}")


# ---------------------------------------------------------------------- main
def zaman_sn(z):
    return int(z[:-1]) * {"m": 60, "h": 3600, "d": 86400}[z[-1]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--semboller", nargs="+", default=["BTC/USDT", "ETH/USDT"])
    ap.add_argument("--zaman", default="1h")
    ap.add_argument("--borsa", default="binance")
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--tek", action="store_true", help="bir kez çalış ve çık (zamanlayıcı için)")
    ap.add_argument("--pencere", type=int, default=20, help="--tek modunda kaç dk içindeki mum sayılsın")
    a = ap.parse_args()

    if a.demo:
        df = sahte_veri()
        d = hesapla(df)
        print("Demo sinyal sayısı  AL:", (d.sinyal == 1).sum(), " SAT:", (d.sinyal == -1).sum())
        print("Backtest:", backtest(df))
        son = d[d.sinyal != 0].iloc[-1]
        print(mesaj_olustur("DEMO/USDT", "4h", son))
        return

    import ccxt
    borsa = getattr(ccxt, a.borsa)({"enableRateLimit": True})

    if a.backtest:
        for s in a.semboller:
            print(s, a.zaman, "->", backtest(veri_cek(borsa, s, a.zaman, 1000)))
        return

    if a.tek:
        # Zamanlayıcıyla (GitHub Actions vb.) çalışır: bir kez bak, çık.
        # Sadece son --pencere dakika içinde kapanan mumdaki sinyali gönderir (tekrar uyarı olmaz).
        simdi = pd.Timestamp.now(tz="UTC").tz_localize(None)
        for s in a.semboller:
            try:
                d = hesapla(veri_cek(borsa, s, a.zaman))
                k = d.iloc[-2]
                kapanis = k.name + pd.Timedelta(seconds=zaman_sn(a.zaman))
                yas = (simdi - kapanis) / pd.Timedelta(minutes=1)
                print(f"{s}: son kapanmış mum {k.name}, yaş {yas:.0f} dk, sinyal {int(k['sinyal'])}")
                if k["sinyal"] != 0 and 0 <= yas <= a.pencere:
                    bildir(mesaj_olustur(s, a.zaman, k))
            except Exception as e:
                print(s, "hata:", e)
        return

    gonderilen = {}
    bildir(f"Bot başladı: {a.semboller} {a.zaman}")
    while True:
        for s in a.semboller:
            try:
                d = hesapla(veri_cek(borsa, s, a.zaman))
                kapanmis = d.iloc[-2]          # son mum henüz kapanmadı, atla
                if kapanmis["sinyal"] != 0 and gonderilen.get(s) != kapanmis.name:
                    bildir(mesaj_olustur(s, a.zaman, kapanmis))
                    gonderilen[s] = kapanmis.name
            except Exception as e:
                print(s, "hata:", e)
        time.sleep(min(zaman_sn(a.zaman) // 4, 300))


if __name__ == "__main__":
    main()
