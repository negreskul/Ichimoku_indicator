import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('seaborn')
import matplotlib.dates as mdates
from mpl_finance import candlestick_ohlc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

class IndicatorMixin:

    _fillna = False

    def _check_fillna(self, series: pd.Series, value: int = 0) -> pd.Series:
        
        if self._fillna:
            series_output = series.copy(deep=False)
            series_output = series_output.replace([np.inf, -np.inf], np.nan)
            if isinstance(value, int) and value == -1:
                series = series_output.fillna(method="ffill").fillna(value=-1)
            else:
                series = series_output.fillna(method="ffill").fillna(value)
        return series

    @staticmethod
    def _true_range(
        high: pd.Series, low: pd.Series, prev_close: pd.Series
    ) -> pd.Series:
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.DataFrame(data={"tr1": tr1, "tr2": tr2, "tr3": tr3}).max(axis=1)
        return true_range

class IchimokuIndicator(IndicatorMixin):
    """ Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        window1(int): n1 low period.
        window2(int): n2 medium period.
        window3(int): n3 high period.
        visual(bool): if True, shift n2 values.
        fillna(bool): if True, fill nan values.
    """

    def __init__(
        self,
        high: pd.Series,
        low: pd.Series,
        window1: int = 9,
        window2: int = 26,
        window3: int = 52,
        visual: bool = False,
        fillna: bool = False,
    ):
        self._high = high
        self._low = low
        self._window1 = window1
        self._window2 = window2
        self._window3 = window3
        self._visual = visual
        self._fillna = fillna
        self._run()

    def _run(self):
        min_periods_n1 = 0 if self._fillna else self._window1
        min_periods_n2 = 0 if self._fillna else self._window2
        self._conv = 0.5 * (
            self._high.rolling(self._window1, min_periods=min_periods_n1).max()
            + self._low.rolling(self._window1, min_periods=min_periods_n1).min()
        )
        self._base = 0.5 * (
            self._high.rolling(self._window2, min_periods=min_periods_n2).max()
            + self._low.rolling(self._window2, min_periods=min_periods_n2).min()
        )

    def ichimoku_conversion_line(self) -> pd.Series:
        """Tenkan-sen (Conversion Line)

        Returns:
            pandas.Series: New feature generated.
        """
        conversion = self._check_fillna(self._conv, value=-1)
        return pd.Series(
            conversion, name=f"ichimoku_conv_{self._window1}_{self._window2}"
        )

    def ichimoku_base_line(self) -> pd.Series:
        """Kijun-sen (Base Line)

        Returns:
            pandas.Series: New feature generated.
        """
        base = self._check_fillna(self._base, value=-1)
        return pd.Series(base, name=f"ichimoku_base_{self._window1}_{self._window2}")

    def ichimoku_a(self) -> pd.Series:
        """Senkou Span A (Leading Span A)

        Returns:
            pandas.Series: New feature generated.
        """
        spana = 0.5 * (self._conv + self._base)
        spana = (
            spana.shift(self._window2, fill_value=spana.mean())
            if self._visual
            else spana
        )
        spana = self._check_fillna(spana, value=-1)
        return pd.Series(spana, name=f"ichimoku_a_{self._window1}_{self._window2}")

    def ichimoku_b(self) -> pd.Series:
        """Senkou Span B (Leading Span B)

        Returns:
            pandas.Series: New feature generated.
        """
        spanb = 0.5 * (
            self._high.rolling(self._window3, min_periods=0).max()
            + self._low.rolling(self._window3, min_periods=0).min()
        )
        spanb = (
            spanb.shift(self._window2, fill_value=spanb.mean())
            if self._visual
            else spanb
        )
        spanb = self._check_fillna(spanb, value=-1)
        return pd.Series(spanb, name=f"ichimoku_b_{self._window1}_{self._window2}")

def add_ichimoku(
    df: pd.DataFrame,
    high: str,
    low: str,
    close: str,
    fillna: bool = False,
    colprefix: str = "",
) -> pd.DataFrame:
    """Add trend technical analysis features to dataframe.

    Args:
        df (pandas.core.frame.DataFrame): Dataframe base.
        high (str): Name of 'high' column.
        low (str): Name of 'low' column.
        close (str): Name of 'close' column.
        fillna(bool): if True, fill nan values.
        colprefix(str): Prefix column names inserted

    Returns:
        pandas.core.frame.DataFrame: Dataframe with new features.
    """
    # Ichimoku Indicator
    indicator_ichi = IchimokuIndicator(
        high=df[high],
        low=df[low],
        window1=9,
        window2=26,
        window3=52,
        visual=False,
        fillna=fillna,
    )
    df[f"{colprefix}trend_ichimoku_conv"] = indicator_ichi.ichimoku_conversion_line()
    df[f"{colprefix}trend_ichimoku_base"] = indicator_ichi.ichimoku_base_line()
    df[f"{colprefix}trend_ichimoku_a"] = indicator_ichi.ichimoku_a()
    df[f"{colprefix}trend_ichimoku_b"] = indicator_ichi.ichimoku_b()
    indicator_ichi_visual = IchimokuIndicator(
        high=df[high],
        low=df[low],
        window1=9,
        window2=26,
        window3=52,
        visual=True,
        fillna=fillna,
    )
    df[f"{colprefix}trend_visual_ichimoku_a"] = indicator_ichi_visual.ichimoku_a()
    df[f"{colprefix}trend_visual_ichimoku_b"] = indicator_ichi_visual.ichimoku_b()

    return df
########################################

df = pd.read_csv("ETH.csv", sep=",")
df = df.append(pd.DataFrame({'Date': pd.date_range(start=df.iloc[df.Date.count()-1].Date, periods=26)}))
df = add_ichimoku(df, "High", "Low", "Close", fillna=True)

df['Date'] =pd.to_datetime(df.Date)

ohlc = df.loc[:, ['Date', 'Open', 'High', 'Low', 'Close']]
ohlc['Date'] = pd.to_datetime(ohlc['Date'])
ohlc['Date'] = ohlc['Date'].apply(mdates.date2num)
ohlc = ohlc.astype(float)

# Random Forest
df4rf = pd.read_csv('CryptoCoinClose.csv')

features = ['BTC', 'XRP', 'XLM', 'LTC' ]
X = df4rf[features]
y = df4rf['ETH']

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

rf_model = RandomForestRegressor(random_state=1, n_estimators = 300, max_depth=15)

rf_model.fit(train_X, train_y)

rf_pred = rf_model.predict(val_X)
date_array = pd.date_range(start="06/04/2021", end="06/13/2021")
predict_array = rf_model.predict(X.tail(10))
print(predict_array.size)
print(date_array.size)

data = { 'Date': date_array,
        'Forecast': predict_array }

df4rf = pd.DataFrame(data)
# Creating Subplots
fig, ax = plt.subplots()

candlestick_ohlc(ax, ohlc.values, width=0.6, colorup='green', colordown='red', alpha=0.8)

plt.plot(df.Date.iloc[:df.Date.count()-26],df.trend_ichimoku_conv.iloc[:df.Date.count()-26], label='Tenkan', color='green')
plt.plot(df.Date.iloc[:df.Date.count()-26],df.trend_ichimoku_base.iloc[:df.Date.count()-26], label='Kijun', color='red')
plt.plot(df.Date,df.trend_ichimoku_a, label='Senkou Span a', color='dimgrey')
plt.plot(df.Date,df.trend_ichimoku_b, label='Senkou Span b', color='dimgray')
#plt.plot(df4rf.Date, df4rf.Forecast, label='Random Forest', color='blue')
plt.fill_between(df.Date,df.trend_ichimoku_a, df.trend_ichimoku_b, where=(df.trend_ichimoku_a > df.trend_ichimoku_b), facecolor='lightgreen')
plt.fill_between(df.Date,df.trend_ichimoku_a, df.trend_ichimoku_b, where=(df.trend_ichimoku_b > df.trend_ichimoku_a), facecolor='tomato')
# Setting labels & titles
ax.set_xlabel('Date')
ax.set_ylabel('Price')
fig.suptitle('Daily Candlestick & Ichimoku Indicators')
plt.legend()
# Formatting Date
date_format = mdates.DateFormatter('%d-%m-%Y')
ax.xaxis.set_major_formatter(date_format)
fig.autofmt_xdate()

fig.tight_layout()

plt.show()