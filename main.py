import datetime as dt
from data_pull.berg import Bbg
from utils import pdf, format_tsy_price


if __name__ == '__main__':
    bbg = Bbg()

    # 1) Reference snapshot
    ref = bbg.bdp(["IBM US Equity", "SPY US Equity"], ["PX_LAST", "CUR_MKT_CAP"])
    pdf(ref)

    # 2) Bulk field
    # members = bbg.bds(["SPX Index"], "INDX_MEMBERS")  # dict: security -> DataFrame
    # print(members)

    # 3) Historical
    hist = bbg.bdh(["TY1 Comdty"], ["PX_LAST"], start="20240101", end="20241031")
    print(hist)

    # 4) Intraday bars
    endt = dt.datetime.now().replace(microsecond=0)
    startt = endt - dt.timedelta(hours=2)
    bars = bbg.intraday_bars("TYA Comdty", startt, endt, event_type="TRADE", interval_minutes=1)
    pdf(bars)

    # 5) Intraday ticks
    # ticks = bbg.intraday_ticks("SPY US Equity", startt, endt, event_types=["TRADE", "BID", "ASK"])
    # pdf(ticks)

    # 6) Discovery
    # hits = bbg.security_search("UST 10Y")
    # finfo = bbg.field_info(["PX_LAST", "PX_OPEN"])
    # pdf(hits)
    # print(finfo)

    # 7) Live Treasury futures feed
    last_print_time = [dt.datetime.min]
    cache = {}

    # Configure which securities use tick notation
    TICK_SECURITIES = {
        "TY1 Comdty": True,  # 10Y Note futures - uses ticks
        "TU1 Comdty": True,  # 2Y Note futures - uses ticks
        "FV1 Comdty": True,  # 5Y Note futures - uses ticks
        # Add others as needed
    }

    def on_tick(sec, data, ts):
        # Update cache with latest data
        if sec not in cache:
            cache[sec] = {}
        cache[sec].update(data)
        cache[sec]['timestamp'] = ts

        # Print throttled to 1 second intervals
        now = dt.datetime.now()
        if (now - last_print_time[0]).total_seconds() >= 1.0:
            last_print_time[0] = now

            for security, values in cache.items():
                parts = [f"{values.get('timestamp', '')} | {security}"]

                # Determine if this security uses tick notation
                use_ticks = TICK_SECURITIES.get(security, False)

                # Format prices appropriately
                if 'LAST_PRICE' in values:
                    formatted = format_tsy_price(values['LAST_PRICE'], is_ticks=use_ticks)
                    parts.append(f"Last: {formatted}")
                if 'BID' in values:
                    formatted = format_tsy_price(values['BID'], is_ticks=use_ticks)
                    parts.append(f"Bid: {formatted}")
                if 'ASK' in values:
                    formatted = format_tsy_price(values['ASK'], is_ticks=use_ticks)
                    parts.append(f"Ask: {formatted}")
                if 'BID_SIZE' in values:
                    parts.append(f"BidSz: {values['BID_SIZE']}")
                if 'ASK_SIZE' in values:
                    parts.append(f"AskSz: {values['ASK_SIZE']}")

                print(" | ".join(parts))


    # Subscribe to Treasury futures
    feed = bbg.subscribe(
        ["WN1 Comdty","TU1 Comdty"],  # 10Y Note futures
        ["LAST_PRICE", "BID", "ASK", "BID_SIZE", "ASK_SIZE"],
        on_data=on_tick
    )

    with feed:
        import time

        time.sleep(5)  # keep main thread alive briefly
