import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

st.set_page_config(page_title='Portfolio Diversification Stress Tester', page_icon='📊', layout='wide')
st.title('Portfolio Diversification Stress Tester')
st.caption('Explore portfolio concentration, return correlations, volatility, and historical drawdowns. Educational use only.')

TRADING_DAYS = 252
STRESS_WINDOWS = {
    'COVID-19 selloff (Feb–Mar 2020)': ('2020-02-19', '2020-03-23'),
    '2022 bear market (Jan–Oct 2022)': ('2022-01-03', '2022-10-12'),
    'Custom range': None,
}

@st.cache_data(ttl=3600, show_spinner=False)
def download_prices(tickers, start, end):
    raw = yf.download(list(tickers), start=start, end=end, auto_adjust=True, progress=False, threads=False)
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        # Supports current yfinance's price-field-first column arrangement.
        prices = raw['Close'] if 'Close' in raw.columns.get_level_values(0) else raw.xs('Close', axis=1, level=1)
    else:
        prices = raw[['Close']].rename(columns={'Close': tickers[0]})
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])
    return prices.reindex(columns=list(tickers)).apply(pd.to_numeric, errors='coerce')

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_sectors(tickers):
    sectors = {}
    for ticker in tickers:
        try:
            sectors[ticker] = yf.Ticker(ticker).info.get('sector') or 'Unknown'
        except Exception:
            sectors[ticker] = 'Unknown'
    return sectors

with st.sidebar:
    st.header('Portfolio setup')
    ticker_text = st.text_input('Stock tickers (comma-separated)', 'AAPL, MSFT, JNJ, JPM')
    tickers = list(dict.fromkeys(x.strip().upper() for x in ticker_text.split(',') if x.strip()))
    if not 2 <= len(tickers) <= 12:
        st.error('Enter 2 to 12 distinct stock tickers.')
        st.stop()
    st.caption('Adjust weights below; values are normalized to 100%.')
    raw_weights = {}
    for ticker in tickers:
        raw_weights[ticker] = st.number_input(f'{ticker} weight', min_value=0.0, max_value=100.0,
                                              value=round(100 / len(tickers), 2), step=1.0, key=f'weight_{ticker}')
    if sum(raw_weights.values()) <= 0:
        st.error('At least one stock must have a positive weight.')
        st.stop()
    weights = pd.Series(raw_weights, dtype=float)
    weights /= weights.sum()
    lookback = st.selectbox('Analysis period', ['2y', '5y', '10y'], index=1)
    benchmark = st.text_input('Benchmark for beta', value='SPY').strip().upper()
    if not benchmark:
        st.error('Enter a benchmark ticker.')
        st.stop()
    st.divider()
    st.subheader('Optional WRDS sector mapping')
    st.caption('Upload a CSV with columns ticker,sector. Do not upload restricted WRDS records to a public repository.')
    sector_file = st.file_uploader('Sector mapping CSV', type=['csv'])

end_date = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
start_date = end_date - pd.DateOffset(years=int(lookback[:-1]))
# Fetch enough history for portfolio analysis and historical stress tests independently.
with st.spinner('Downloading historical prices...'):
    prices = download_prices(tuple(tickers + ([benchmark] if benchmark not in tickers else [])),
                             start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
if prices.empty:
    st.error('No price data returned. Check the ticker symbols or retry later.')
    st.stop()

missing = [ticker for ticker in tickers if ticker not in prices or prices[ticker].dropna().empty]
if missing:
    st.error('No data found for: ' + ', '.join(missing))
    st.stop()

# Use only common observed dates: never forward-fill missing historical prices into returns.
asset_prices = prices[tickers].dropna(how='any')
if len(asset_prices) < 65:
    st.error('Not enough overlapping price history. Try different tickers.')
    st.stop()
returns = asset_prices.pct_change().dropna()
w = weights.reindex(tickers).to_numpy()
covariance = returns.cov().to_numpy() * TRADING_DAYS
portfolio_vol = float(np.sqrt(max(w @ covariance @ w, 0)))
portfolio_daily = returns @ w   # constant-weight daily-rebalanced portfolio
portfolio_growth = (1 + portfolio_daily).cumprod()
annual_return = float((portfolio_growth.iloc[-1] ** (TRADING_DAYS / len(portfolio_daily))) - 1)
stock_vols = pd.Series(np.sqrt(np.diag(covariance)), index=tickers)
if portfolio_vol > 0:
    component_risk = pd.Series(w * (covariance @ w) / portfolio_vol, index=tickers)
    risk_share = component_risk / portfolio_vol
else:
    risk_share = pd.Series(0.0, index=tickers)
drawdown = portfolio_growth / portfolio_growth.cummax() - 1

benchmark_prices = prices[benchmark].dropna() if benchmark in prices else pd.Series(dtype=float)
benchmark_returns = benchmark_prices.pct_change().dropna()
common = pd.concat([portfolio_daily.rename('portfolio'), benchmark_returns.rename('benchmark')], axis=1).dropna()
beta = (common['portfolio'].cov(common['benchmark']) / common['benchmark'].var()) if len(common) > 10 and common['benchmark'].var() > 0 else np.nan

st.caption(f'Analysis uses {asset_prices.index.min():%b %d, %Y} to {asset_prices.index.max():%b %d, %Y} • All returns are historical, not forecasts.')
a,b,c,d = st.columns(4)
a.metric('Annualized volatility', f'{portfolio_vol:.1%}')
b.metric('Annualized historical return', f'{annual_return:.1%}')
c.metric('Maximum drawdown', f'{drawdown.min():.1%}')
d.metric(f'Beta vs {benchmark}', f'{beta:.2f}' if np.isfinite(beta) else 'N/A')

risk_tab, diversification_tab, stress_tab, data_tab = st.tabs(['Risk overview', 'Diversification & sectors', 'Stress testing', 'Data & methodology'])
with risk_tab:
    left, right = st.columns(2)
    with left:
        st.subheader('Portfolio allocation')
        fig = px.pie(values=weights.values, names=weights.index, hole=.45)
        st.plotly_chart(fig, width='stretch')
    with right:
        st.subheader('Contribution to total portfolio risk')
        chart = risk_share.sort_values().rename('Risk contribution').reset_index().rename(columns={'index': 'Ticker'})
        fig = px.bar(chart, x='Risk contribution', y='Ticker', orientation='h', text='Risk contribution')
        fig.update_traces(texttemplate='%{text:.1%}', textposition='outside')
        fig.update_xaxes(tickformat='.0%')
        st.plotly_chart(fig, width='stretch')
    st.info('Risk contributions sum to 100%. A negative contribution means a holding reduces overall variance in this historical sample.')
    st.subheader('Stock-level risk')
    stats = pd.DataFrame({'Portfolio weight': weights, 'Annualized volatility': stock_vols,
                          'Share of portfolio volatility': risk_share})
    st.dataframe(stats.style.format('{:.1%}'), width='stretch')

with diversification_tab:
    corr = returns.corr()
    st.subheader('Return correlation heatmap')
    st.plotly_chart(px.imshow(corr, zmin=-1, zmax=1, color_continuous_scale='RdBu_r',
                              text_auto='.2f', aspect='auto'), width='stretch')
    st.subheader('Hierarchical clustering (machine learning)')
    st.caption('Stocks are grouped by similarity in daily return correlations; this is unsupervised clustering, not a prediction.')
    if len(tickers) > 2:
        # Distance sqrt(2(1-rho)); average-linkage agglomerative hierarchical clustering.
        dist = np.sqrt(np.maximum(0, 2 * (1 - corr.to_numpy())))
        np.fill_diagonal(dist, 0)
        tree = linkage(squareform(dist, checks=False), method='average')
        ordered = [tickers[i] for i in leaves_list(tree)]
        st.plotly_chart(px.imshow(corr.loc[ordered, ordered], zmin=-1, zmax=1,
                                  color_continuous_scale='RdBu_r', text_auto='.2f', aspect='auto'),
                        width='stretch')
        st.caption('The reordered matrix places more similarly behaving stocks together.')
    else:
        st.caption('Add a third stock to see a meaningful cluster arrangement.')
    st.subheader('Sector allocation')
    mapping = fetch_sectors(tuple(tickers)) if sector_file is None else {t: 'Unknown' for t in tickers}
    if sector_file is not None:
        try:
            sector_csv = pd.read_csv(io.BytesIO(sector_file.getvalue()))
            sector_csv.columns = sector_csv.columns.str.strip().str.lower()
            if not {'ticker', 'sector'}.issubset(sector_csv.columns):
                st.warning('CSV needs ticker and sector columns. Unknown sectors will be shown.')
            else:
                mapping.update(dict(zip(sector_csv['ticker'].astype(str).str.upper().str.strip(),
                                        sector_csv['sector'].astype(str))))
        except Exception as exc:
            st.warning(f'Unable to read sector CSV: {exc}')
    sector_weights = pd.DataFrame({'sector': [mapping.get(t, 'Unknown') for t in tickers],
                                   'weight': weights.values}).groupby('sector', as_index=False)['weight'].sum()
    st.plotly_chart(px.bar(sector_weights, x='sector', y='weight', text='weight',
                           labels={'sector':'Sector', 'weight':'Portfolio weight'}).update_traces(texttemplate='%{text:.1%}'),
                    width='stretch')
    st.caption('Sector concentration is descriptive; cross-sector correlations also matter.')

with stress_tab:
    st.subheader('Historical market downturn simulation')
    scenario = st.selectbox('Choose scenario', list(STRESS_WINDOWS))
    if scenario == 'Custom range':
        start_stress = st.date_input('Start date', value=pd.Timestamp('2022-01-03').date())
        end_stress = st.date_input('End date', value=pd.Timestamp('2022-10-12').date())
        stress_start, stress_end = str(start_stress), str(end_stress)
    else:
        stress_start, stress_end = STRESS_WINDOWS[scenario]
    # Download the scenario independently so the lookback selector does not silently truncate it.
    with st.spinner('Loading stress-window history...'):
        stress_prices = download_prices(tuple(tickers), stress_start,
                                        (pd.Timestamp(stress_end) + pd.Timedelta(days=1)).strftime('%Y-%m-%d'))
    stress_prices = stress_prices.reindex(columns=tickers).dropna(how='any')
    if len(stress_prices) < 3:
        st.warning('This scenario has insufficient overlapping historical prices for the selected stocks. Try a later period or different tickers.')
    else:
        sr = stress_prices.pct_change().dropna()
        sp = sr @ w
        # Initial level = 1 before first return; constant-weight daily rebalancing.
        performance = pd.concat([pd.Series([1.0], index=[stress_prices.index[0]]), (1+sp).cumprod()])
        performance = performance[~performance.index.duplicated(keep='last')]
        scenario_return = performance.iloc[-1] - 1
        scenario_drawdown = (performance / performance.cummax() - 1).min()
        x,y = st.columns(2)
        x.metric('Return over selected period', f'{scenario_return:.1%}')
        y.metric('Worst drawdown within period', f'{scenario_drawdown:.1%}')
        fig = px.line(performance.rename('Portfolio growth').reset_index().rename(columns={'index':'Date'}),
                      x='Date', y='Portfolio growth', title='Growth of $1 invested at start')
        st.plotly_chart(fig, width='stretch')
        individual = ((1 + sr).prod() - 1).rename('Historical return').sort_values()
        st.plotly_chart(px.bar(x=individual.index, y=individual.values,
                               labels={'x':'Ticker', 'y':'Historical return'}, title='Stock returns during scenario'), width='stretch')
    st.caption('Simulation assumes daily rebalancing to selected weights, no trading costs, and no taxes. It is a historical scenario, not a forecast.')

with data_tab:
    st.markdown('''**Data:** Yahoo Finance historical adjusted prices via `yfinance`, updated when the 1-hour cache expires. Optional WRDS-derived sector mapping CSV.\n\n**Portfolio model:** Daily simple returns; annualized covariance = daily covariance × 252; annualized volatility = square root of weighted covariance. Component risk contribution uses each holding’s covariance with the portfolio, weighted and scaled by total volatility.\n\n**Beta:** Covariance of portfolio and benchmark daily returns divided by benchmark daily return variance.\n\n**Clustering:** Average-linkage agglomerative hierarchical clustering using correlation distance `sqrt(2 × (1 − correlation))`.\n\n**Limitations:** Historical relationships can change. Stocks with missing overlapping price history may be excluded from analysis dates; survivorship bias and data-vendor gaps are possible. Do not treat these metrics as investment advice.''')
    st.download_button('Download analyzed daily returns (CSV)', returns.to_csv().encode('utf-8'),
                       file_name='portfolio_daily_returns.csv', mime='text/csv')
