from tabulate import tabulate, tabulate_formats


def pdf(df):
    print(tabulate(df, headers='keys', tablefmt=tabulate_formats[2]))


def format_tsy_price(decimal_price, is_ticks=True):
    """
    Convert decimal price to Treasury futures notation.

    Examples:
        112.64062500 -> 112-20+
        112.625 -> 112-20  (if is_ticks=False)
        112.640625 -> 112-20+ (20.5/32)
        112.65625 -> 112-21  (21/32)
    """
    # Split into handles and fractional part
    handles = int(decimal_price)
    frac = decimal_price - handles

    if is_ticks:
        # Price is in ticks where 1 tick = 1/256 (32nds of 32nds, or 1/(32*8))
        # Convert to 32nds
        ticks = frac * 256  # total ticks
        thirty_seconds = ticks / 8  # convert to 32nds
    else:
        # Price already in 32nds (fractional part represents 32nds directly)
        thirty_seconds = frac * 32

    # Round to nearest 1/8th of a 32nd
    eighth_of_32nd = round(thirty_seconds * 8) / 8

    # Split into whole 32nds and the fraction
    whole_32nds = int(eighth_of_32nd)
    remainder = eighth_of_32nd - whole_32nds

    # Format the fractional part
    if abs(remainder - 0.5) < 0.001:  # it's a half (4/8)
        suffix = "+"
    elif abs(remainder - 0.25) < 0.001:  # it's 2/8
        suffix = "2"
    elif abs(remainder - 0.75) < 0.001:  # it's 6/8
        suffix = "6"
    elif abs(remainder) < 0.001:  # it's 0/8
        suffix = ""
    else:
        # Shouldn't happen after rounding to 1/8th, but just in case
        suffix = f"{int(remainder * 8)}"

    return f"{handles:03d}-{whole_32nds:02d}{suffix}"