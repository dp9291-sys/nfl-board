# NFL ATS Board

Live page: https://dp9291-sys.github.io/nfl-board/

A personal against-the-spread board. Open it on your phone, enter your Odds API key once (stored only in that
browser), and press **Refresh lines** to price every game at its best available book. Weather refreshes itself.

**Model:** opponent-adjusted EPA ratings, QB values, injuries, schedule spots, and the one signal that beat
closing lines in walk-forward testing (special teams), combined with key-number cover odds at posted prices.
Tiers from the 2018-2026 backtest: Tier A (EV 5%+) 56.4% / +11.6% ROI; Tier B (EV 3-5%) 52.0% / +3.1%.

**Weekly rebuild:** `pip install -r requirements.txt && python build_site.py`, then commit `index.html`.
Code in `nflats/` (data from nflverse). No keys or personal data are stored in this repository.
