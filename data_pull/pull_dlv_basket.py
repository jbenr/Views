#!/usr/bin/env python3
"""
Populate deliverable basket data for Treasury futures contracts.
Fetches FUT_DLVRBLE_BNDS_CUSIPS from Bloomberg and stores in sec.fut_contracts.
"""

import os
import json
import argparse
import psycopg
import sys
from tqdm import tqdm

sys.path.append(os.path.expanduser("~/werk/Views"))
from data_pull.berg import Bbg

DB_DSN = os.getenv("DB_DSN", "postgresql://benjils:snickers@raptor:5432/markets")
BATCH_SIZE = 50
DELIVERABLE_ROOTS = ["TU", "FV", "TY", "UXY", "US", "WN", "UB", "Z3N"]


def ensure_dlv_basket_column(conn):
    """Add dlv_basket column if it doesn't exist."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_schema = 'sec' 
              AND table_name = 'fut_contracts' 
              AND column_name = 'dlv_basket'
        """)
        if cur.fetchone() is None:
            print("Adding dlv_basket column...")
            cur.execute("ALTER TABLE sec.fut_contracts ADD COLUMN dlv_basket JSONB")
            conn.commit()
            print("Column added.")
        else:
            print("dlv_basket column already exists.")


def get_contracts(conn, only_missing: bool = True):
    """Get contracts to update."""
    with conn.cursor() as cur:
        if only_missing:
            cur.execute("""
                SELECT contract
                FROM sec.fut_contracts
                WHERE dlv_basket IS NULL
                  AND generic_ticker = ANY(%s)
                ORDER BY contract_month DESC
            """, (DELIVERABLE_ROOTS,))
        else:
            cur.execute("""
                SELECT contract
                FROM sec.fut_contracts
                WHERE generic_ticker = ANY(%s)
                ORDER BY contract_month DESC
            """, (DELIVERABLE_ROOTS,))
        return [row[0] for row in cur.fetchall()]


def fetch_deliverable_baskets(bbg: Bbg, contracts: list[str]) -> dict[str, list[dict]]:
    """
    Fetch deliverable baskets from Bloomberg for multiple contracts.
    Returns dict: contract -> list of {cusip, conversion_factor}
    """
    tickers = [f"{c} Comdty" for c in contracts]
    
    # bds returns dict: ticker -> DataFrame
    result = bbg.bds(tickers, "FUT_DLVRBLE_BNDS_CUSIPS")
    
    out = {}
    for contract, ticker in zip(contracts, tickers):
        if ticker not in result:
            out[contract] = None
            continue
            
        df = result[ticker]
        
        if df.empty or 'error' in df.columns:
            out[contract] = None
            continue
        
        basket = []
        for _, row in df.iterrows():
            # Use positional indexing since column names may vary
            cusip_raw = row.iloc[0] if len(row) > 0 else ''
            cf = row.iloc[1] if len(row) > 1 else 0
            
            # Extract just the CUSIP (remove " Govt" suffix)
            cusip = str(cusip_raw).replace(' Govt', '').strip()
            
            basket.append({
                'cusip': cusip,
                'conversion_factor': float(cf)
            })
        
        out[contract] = basket if basket else None
    
    return out


def update_dlv_basket(conn, contract: str, basket: list[dict]):
    """Update the dlv_basket column for a contract."""
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE sec.fut_contracts 
            SET dlv_basket = %s, updated_at = NOW()
            WHERE contract = %s
        """, (json.dumps({'deliverables': basket}), contract))
    conn.commit()


def fetch_and_store_baskets(
    bbg: Bbg,
    conn,
    contracts: list[str],
    batch_size: int = BATCH_SIZE,
) -> tuple[int, int]:
    """
    Fetch deliverable baskets from Bloomberg in batches and store.
    Returns (success_count, no_data_count).
    """
    success = 0
    no_data = 0
    n_batches = (len(contracts) + batch_size - 1) // batch_size

    for i in tqdm(
        range(0, len(contracts), batch_size),
        total=n_batches,
        desc="Bloomberg BDS batches",
        unit="batch",
    ):
        batch = contracts[i : i + batch_size]
        baskets = fetch_deliverable_baskets(bbg, batch)

        for contract, basket in baskets.items():
            if basket is None:
                no_data += 1
            else:
                update_dlv_basket(conn, contract, basket)
                success += 1

    return success, no_data


def main():
    parser = argparse.ArgumentParser(description='Populate deliverable basket data for futures contracts')
    parser.add_argument('--all', action='store_true', help='Update all contracts, not just missing ones')
    args = parser.parse_args()
    
    print(f"Connecting to Postgres: {DB_DSN}")
    conn = psycopg.connect(DB_DSN)
    
    # Step 1: Ensure column exists
    ensure_dlv_basket_column(conn)
    
    # Step 2: Get contracts to update
    contracts = get_contracts(conn, only_missing=not args.all)
    
    if args.all:
        print(f"Found {len(contracts)} total contracts to update.")
    else:
        print(f"Found {len(contracts)} contracts missing deliverable basket data.")
    
    if not contracts:
        print("Nothing to do.")
        conn.close()
        return
    
    # Step 3: Initialize Bloomberg
    print("Connecting to Bloomberg...")
    bbg = Bbg()
    
    # Step 4: Fetch and store
    success, no_data = fetch_and_store_baskets(bbg, conn, contracts)
    
    print(f"\n✅ Done. Success: {success}, No data: {no_data}")
    conn.close()


if __name__ == "__main__":
    main()
