import pandas as pd
import os
import random
from threading import Thread
import time
from typing import Optional, List
import requests
from requests.exceptions import HTTPError
import plotly.express as px
import plotly.graph_objects as go
from solara.alias import rv
from datetime import datetime
from openai import OpenAI, APIConnectionError
import solara as sl
from pydantic_settings import BaseSettings, SettingsConfigDict
from urllib.parse import urlencode
from loader import Loader

TITLE = "Dividend Stock Analyzer"

DATA_PROMPT = """\
{question}

Company: {company}
Ticker: {ticker}

Earnings:

{earnings}

DIVIDENDS:

{dividends}

P/E:

{pe}

DEBT/EQUITY:

{dte}

CASH/SHARE:

{cps}

FREE CASH FLOW/SHARE:

{fcf}

PAYOUT RATIO:

{payout}
"""

class Envs(BaseSettings):
    fmp_api_key: str
    openai_api_key: str
    openai_model: str = "gpt-3.5-turbo"

kwargs = {} if os.getenv("IN_DOCKER") == "true" else dict(_env_file="../../.env")
envs = Envs(**kwargs)

TIME_DIFFS = {
    "1 week": pd.DateOffset(weeks=1),
    "1 month": pd.DateOffset(months=1),
    "3 months": pd.DateOffset(months=3),
    "1 year": pd.DateOffset(years=1),
    "3 years": pd.DateOffset(years=3),
    "5 years": pd.DateOffset(years=5)
}

def get_price_data_fig(srs, moving_average, time_window, time_window_key, currency):
    # create moving average
    ma = srs.rolling(window=moving_average).mean().dropna()
    # only in time window
    start = (pd.to_datetime("today").floor("D") - time_window)
    srs = srs.loc[start:]
    ma = ma.loc[start:]
    # create figures for normal and moving average
    fig1 = px.line(y=srs, x=srs.index)
    fig1.update_traces(line_color="blue", name="Price", showlegend=True)
    fig2 = px.line(y=ma, x=ma.index)
    fig2.update_traces(line_color="orange", name=f"Moving average price ({moving_average})", showlegend=True)
    # combine and add layout
    fig = go.Figure(data = fig1.data + fig2.data)
    fig.update_layout(
        title=f"Adjusted closing price last {time_window_key}",
        xaxis_title="Date",
        yaxis_title=currency,
        title_x = 0.5,
        # align labels top-left, side-by-side
        legend=dict(y=1.1, x=0, orientation="h"),
        showlegend=True,
        height=500
    )
    return fig


def plot_data(data, key, title, yaxis_title, show_mean=False, mean_text="", type="line"):
    # getattr(px, type) if type = 'line' is px.line
    fig = getattr(px, type)(y=data[key], x=data[key].index)
    # add a historical mean if specified
    if show_mean:
        fig.add_hline(data[key].mean(), line_dash="dot", annotation_text=mean_text)
    # set title and axis-titles
    fig.update_layout(
        title=title, 
        xaxis_title="Date",
        yaxis_title=yaxis_title,
        title_x = 0.5,
        showlegend=False
    )
    return fig

class FMPAPI:
    def __init__(self, api_key: str, n_requests_in_parallel: Optional[int] = None):
        self._api_key = api_key
        self.data = {}
        self.n_requests_in_parallel = n_requests_in_parallel
        self._threads = []

    def get_sync(self, *args, **kwargs):
        self._get(*args, **kwargs)
        return self.data[args[0]]

    def get(self, *args, **kwargs):
        if self.n_requests_in_parallel is not None and len(self._threads) >= self.n_requests_in_parallel:
            self.collect(1)
        th = Thread(target=self._get, args=args, kwargs=kwargs)
        self._threads.append(th)
        th.start()

    def collect(self, n: Optional[int] = None):
        if n is None:
            n = len(self._threads)
        for th in self._threads[:n]:
            th.join()
        self._threads = self._threads[n:]

    def _get(self, name: str, endpoint: str, to_dataframe: bool = True, first: bool = False, use_key: Optional[str]=None, **query_params: str | float | int):
        if query_params is None:
            query_params = {}
        else:
            # handle reserved keywords in Python, e.g. _from -> from
            query_params = { 
                (k if not k.startswith("_") else k[1:]): v 
                for k, v in query_params.items()
            }
        query_params |= { "apikey": self._api_key }
        qs = urlencode(query_params)
        if endpoint.startswith("/"):
            endpoint = endpoint[1:]
        response = requests.get(f"https://financialmodelingprep.com/api/v3/{endpoint}?{qs}")
        response.raise_for_status()

        data = response.json()

        if first:
            data = data[0]

        if use_key is not None:
            data = data[use_key]

        if to_dataframe:
            data = pd.DataFrame(data)
            if "date" in data.columns:
                data = data.set_index("date")
                data.index = pd.to_datetime(data.index)
            data = data.sort_index()
        self.data[name] = data

@sl.component
def AppComponent(title: Optional[str] = None, children: List[sl.Element] = None, **kwargs):
    if title is not None:
        # add it before the other content
        children = [
            sl.HTML("h1", title, style="text-align: center; color: black;"),
            sl.HTML("br")
        ] + ([] if children is None else children)
    sl.Card(children=children, **kwargs)

@sl.component
def ShowMore(text: str, n: int = 400):
    show_more, set_show_more = sl.use_state(False)

    if show_more or len(text) < n:
        sl.Markdown(text)
        if len(text) > n:
            sl.Button("Show less", on_click=lambda: set_show_more(False))
    else:
        sl.Markdown(text[:n] + "...")
        sl.Button("Show more", style="width: 150px;", on_click=lambda: set_show_more(True))

@sl.component
def AIAnalysis(ticker: str | None, data: dict):
    ai_analysis, set_ai_analysis = sl.use_state(None)
    asked_last, set_asked_last = sl.use_state(None)
    question, set_question = sl.use_state("")
    load, set_load = sl.use_state(False)

    def fetch_ai_analysis():
        if not load:
            return

        if asked_last is not None:
            _ticker, _question = asked_last
            # same, don't fetch
            if _ticker == ticker and _question == question:
                return
            set_asked_last((ticker, question))

        client = OpenAI(api_key=envs.openai_api_key)

        # reset
        set_ai_analysis("")

        stream = client.chat.completions.create(
            model=envs.openai_model,
            messages=[
                { 
                    "role": "system", 
                    "content": """
                        You are a Stock Market Analyst, a financial expert deeply versed 
                        in the intricacies of the stock market. Your keen ability to dissect complex
                        market trends, interpret financial data, and understand the implications of 
                        historical data on stock performance makes your insight invaluable. 
                        With a talent for predicting market movements and understanding investment strategies, 
                        you balance an analytical approach with an intuitive understanding of market psychology.
                        Your guidance is sought after for making informed, strategic decisions in the dynamic 
                        world of stock trading.
                    """
                },
                {
                    "role": "user", 
                    "content": DATA_PROMPT.format(
                        ticker=ticker,
                        question=(
                            "Give me your insights into the following stock:" 
                            if question is None or question == ""
                            else question
                        ),
                        company=data["info"]["companyName"],
                        earnings=data["earnings_per_share"].to_string(),
                        dividends=data["dividends"].to_string(),
                        pe=data["historical_PE"].to_string(),
                        dte=data["debt_to_equity"].to_string(),
                        cps=data["cash_per_share"].to_string(),
                        fcf=data["free_cash_flow_per_share"].to_string(),
                        payout=data["payout_ratio"].to_string()
                    )
                }
            ],
            stream=True,
            temperature=0.5
        )
        combined = ""
        for chunk in stream:
            added = chunk.choices[0].delta.content
            if added is not None:
                combined += added
                set_ai_analysis(combined)

        set_load(False)

    # fetch in a separate thread to not block UI
    fetch_thread = sl.use_thread(fetch_ai_analysis, dependencies=[load])

    def stop_fetching():
        fetch_thread.cancel()
        set_load(False)

    if data is None:
        return
    
    with AppComponent("AI Analysis"):
        rv.Textarea(
            v_model=question, 
            on_v_model=set_question,
            outlined=True, 
            hide_details=True,
            label="Question (optional)", 
            rows=3, 
            auto_grow=True
        )
        sl.HTML("br")
        if not load and (asked_last is None or asked_last != (ticker, question)):
            sl.Button("Analyze", icon_name="mdi-auto-fix", color="primary", on_click=lambda: set_load(True))
        else:
            sl.Button("Stop", icon_name="mdi-cancel", color="secondary", on_click=lambda: stop_fetching())
        sl.HTML("br")
        sl.HTML("br")

        if ai_analysis is not None:
            sl.HTML("h3", "Output:")
            ShowMore(ai_analysis)

@sl.component
def PercentageChange(data: Optional[dict]):
    if data is None:
        return
    
    # no title
    with AppComponent():        
        # Add changes for different periods
        close = data["stock_closings"]
        latest_price = close.iloc[-1]# data["info"]["price"]
        # should all be displayed on the same row
        today = pd.to_datetime("today").floor("D")

        with sl.Columns([1]*len(TIME_DIFFS)):
            for name, difference in TIME_DIFFS.items():
                # go back to the date <difference> ago
                date = (today - difference)
                # if there is no data back then, then use the earliest
                if date < close.index[0]:
                    date = close.index[0]
                # if no match, get the date closest to it back in time, e.g. weekend to friday
                idx = close.index.get_indexer([date],method='ffill')[0]
                previous_price = close.iloc[idx]
                # calculate change in percent
                change = 100*(latest_price - previous_price) / previous_price
                # show red if negative, green if positive
                color = "red" if change < 0 else "green"

                with sl.Row():
                    sl.Text(name)
                    sl.Text(f"{round(change, 2)}%", style=dict(color=color))

def Overview(info: Optional[dict]=None):
    # basic information
    if info is None:
        return
    with AppComponent("Overview"):
        for text, key in [
            ("Current price", "price"),
            ("Country", "country"),
            ("Exchange", "exchange"),
            ("Sector", "sector"),
            ("Industry", "industry"),
            ("Full time employees", "fullTimeEmployees")
        ]:
            sl.Markdown(f"- {text}: **{info[key]}**")

@sl.component
def Description(info: dict):
    if info is None:
        return

    with AppComponent("Description"):
        ShowMore(info["description"])

@sl.component
def Price(data: dict):

    moving_average, set_moving_average = sl.use_state(30)
    time_window_key, set_time_window_key = sl.use_state("5 years")
    # select the value from the key, i.e. the pd.DateOffset
    time_window = TIME_DIFFS[time_window_key]

    if data is None:
        return
    
    info = data["info"]

    with AppComponent("Price"):        
        # here I set different widths to each column,
        # meaning the first is 1 width and the second 3,
        # i.e. 1/(1+3) = 25% and 3 / (1+4) = 75%
        with sl.Columns([1, 3], style="align-items: center;"):
        
            # second column, graph and graph settings
            with sl.Column():
                # show the graph
                fig = get_price_data_fig(data["stock_closings"], moving_average, time_window, time_window_key, info["currency"])
                sl.FigurePlotly(fig)

                # options that will dictate the graph:

                # radio buttons for what time window to display the stock price
                with sl.Column(align="center"):
                    sl.HTML("h2", "Time window")
                    sl.ToggleButtonsSingle(
                        value=time_window_key,
                        values=list(TIME_DIFFS.keys()),
                        on_value=set_time_window_key
                    )
                # set moving average
                sl.SliderInt(label=f"Moving average: {moving_average}", min=2, max=500, value=moving_average, on_value=set_moving_average)

@sl.component
def Charts(data: dict):
    if data is None:
        return

    with AppComponent("Charts"):        
        currency = data["info"]["currency"]

        # define all plots
        div_fig = plot_data(
            data,
            key="dividends",
            title="Adjusted dividends",
            yaxis_title=currency,
            type="bar"
        )
        pe_fig = plot_data(
            data, 
            key="historical_PE", 
            title="Historical Price-to-Earnings (P/E) Ratio", 
            yaxis_title="P/E", 
            show_mean=True, 
            mean_text="Average Historical P/E",
        )
        yield_fig = plot_data(
            data, 
            key="dividend_yield", 
            title="Dividend Yield", 
            yaxis_title="Percent %", 
            show_mean=True, 
            mean_text="Average Historical Dividend Yield",
            type="bar"
        )
        payout_fig = plot_data(
            data, 
            key="payout_ratio", 
            title="Payout Ratio", 
            yaxis_title="Payout Ratio",
            type="bar",
            show_mean=True,
            mean_text="Average Historical Payout Ratio"
        )
        cps_fig = plot_data(
            data, 
            key="cash_per_share", 
            title="Cash/Share", 
            yaxis_title=currency,
            type="bar"
        )
        fcf_fig = plot_data(
            data, 
            key="free_cash_flow_per_share", 
            title="Free Cash Flow/Share", 
            yaxis_title=currency,
            type="bar"
        )
        eps_fig = plot_data(
            data, 
            key="earnings_per_share", 
            title="Earnings/Share", 
            yaxis_title=currency,
            type="bar"
        )
        dte_fig = plot_data(
            data,
            key="debt_to_equity",
            title="Debt-to-equity",
            yaxis_title="Debt/Equity"   
        )

        # align plots side by side
        combos = [(div_fig, pe_fig), (eps_fig, yield_fig), (payout_fig, cps_fig), (fcf_fig, dte_fig)]
        for (fig1, fig2) in combos:
            with sl.Columns([1,1]):
                if fig1 is not None:
                    sl.FigurePlotly(fig1)
                if fig2 is not None:
                    sl.FigurePlotly(fig2)

@sl.component
def FunAnimation():
    sl.Style("""
        .loading-text {
            animation: loading-text-animation 3s infinite;
        }

        @keyframes loading-text-animation {
            0% {
                color: #3f00b3;
            }
            50% {
                color: #13bdfc;
            }
            100% {
                color: #3f00b3;
            }
        }         
    """)

    loader, set_loader = sl.use_state("")
    texts = [
        "Consulting with our financial wizards...",
        "Negotiating with the stock market gremlins...",
        "Brewing a fresh pot of financial data..."
    ]
    def animate():
        while True:
            set_loader(random.choice(texts))
            time.sleep(3)
    sl.use_thread(animate, dependencies=[])

    sl.HTML("h3", loader, class_="loading-text")

@sl.component
def Page():
    # css styling
    sl.Style("""
        html, body {
            margin: 10px;
            overflow-x: hidden;
            width: 100%;
        }
        .logo {
            width: 150px;
            margin-top: 10px;
            border-radius: 100%;
            box-shadow: 0 0 3px rgba(0,0,0,0.5);
        }
             
        .logo-animation {
            animation: logo-animation 1s infinite;
        }
             
        @keyframes logo-animation {
            0% {
                box-shadow: 0 0 3px rgba(0,0,0,0.5);
            }
            50% {
                box-shadow: 0 0 10px rgb(0 133 255);
            }
            100% {
                box-shadow: 0 0 3px rgba(0,0,0,0.5);
            }
        }

        /* remove solara watermark */
        .v-application--wrap > :nth-child(2) > :nth-child(2) {
            visibility: hidden;
        }
    """)

    # title, shown in the browser tab
    sl.Title(TITLE)

    # define states

    # two ticker states, one is updated on input,
    # the other after the button is clicked to load the data
    ticker, set_ticker = sl.use_state(None)
    input_ticker, set_input_ticker = sl.use_state("")
    error, set_error = sl.use_state(False)
    # data fetched from FMP API
    data, set_data = sl.use_state(None)

    # method to fetch all data from FMP API
    def fetch_data():
        if ticker is None or ticker == "":
            return
        
        # the cache holds (data, expiration_time)
        if ticker in sl.cache.storage and sl.cache.storage[ticker][1] > datetime.utcnow():
            print(f"Fetching ${ticker} data from cache")
            set_data(sl.cache.storage[ticker][0])
            return
        
        print(f"Fetching ${ticker} data from API")

        # run multiple requests in parallel and collect the results here
        
        api = FMPAPI(api_key=envs.fmp_api_key, n_requests_in_parallel=3)
        # assert it exists
        try:
            api.get_sync("profile", f"/profile/{ticker}", to_dataframe=False, first=True)
            set_error(False)
        except IndexError:
            set_error("Ticker not found")
            set_data(None)
            set_ticker(None)
            return
        except HTTPError as e:
            if e.response.status_code == 401:
                set_error("Invalid API key")
                set_data(None)
                set_ticker(None)
                return
            # unknown error
            raise e

        
        api.get("key_metrics_annually", f"/key-metrics/{ticker}", period="annual")
        # by default 5 years, daily
        api.get("stock_data", f"/historical-price-full/{ticker}", use_key="historical")
        api.get("financial_ratios_annually", f"/ratios/{ticker}", period="annual")
        api.get("income_statement_annually", f"/income-statement/{ticker}", period="annual")
        api.get("dividends", f"/historical-price-full/stock_dividend/{ticker}", use_key="historical")
            
        api.collect()

        profile = api.data["profile"]
        key_metrics_annually = api.data["key_metrics_annually"]
        stock_data = api.data["stock_data"]
        financial_ratios_annually = api.data["financial_ratios_annually"]
        income_statement_annually = api.data["income_statement_annually"]
        dividends = api.data["dividends"]
        try:
            divs = dividends["adjDividend"].resample("1Y").sum().sort_index()
        except KeyError:
            dividends=None
            divs = pd.Series(0, name="Dividends")
        data = {
            "stock_closings": stock_data["adjClose"],
            "historical_PE": key_metrics_annually["peRatio"],
            "payout_ratio": financial_ratios_annually["payoutRatio"],
            "dividend_yield": 100*financial_ratios_annually["dividendYield"],
            "cash_per_share": key_metrics_annually["cashPerShare"],
            "debt_to_equity": key_metrics_annually["debtToEquity"],
            "free_cash_flow_per_share": key_metrics_annually["freeCashFlowPerShare"],
            "dividends": divs,
            "earnings_per_share": income_statement_annually["eps"],
            "info": profile,
            "all": dict(
                key_metrics_annually=key_metrics_annually,
                financial_ratios_annually=financial_ratios_annually,
                income_statement_annually=income_statement_annually,
                dividends=dividends
            )
        }

        # update cache and set expiration time to 1 hour from now
        sl.cache.storage[ticker] = (data, datetime.utcnow() + pd.DateOffset(hours=1))
        # set the `data` state
        set_data(data)

    # fetch in a thread to not block the UI
    sl.use_thread(fetch_data, dependencies=[ticker])

    def update_ticker():
        set_data(None)
        set_ticker(input_ticker)

    # when ticker is changed but not data => loading
    is_loading = data is None and ticker is not None and ticker != ""

    # image and title
    with sl.Column(align="center"):
        # classes is CSS-classes,
        # here an extra animation is added during loading
        sl.Image("./logo.png", classes=["logo"] + ([] if not is_loading else ["logo-animation"]))
        sl.HTML("h1", TITLE)

    # sl.Columns takes a list of widths, [1, 1, 1] means 3 columns with equal width
    with sl.Columns([1, 1, 1]):
        # the first element will have the first width, 
        # the second the second width, etc.

        # sl.HTML("div") is used as a placeholder
        sl.HTML("div")
        with sl.Column(align="center"):
            if not is_loading:
                # $ + input + button
                with sl.Row(style="align-items: center;"):
                    sl.Text("$", style="color: gray")
                    # this input is connected to the state of
                    # `input_ticker`, which will in turn update 
                    # the `ticker` state after clicking the submit button
                    sl.InputText(
                        label="Ticker", 
                        value=input_ticker, 
                        error=error, 
                        continuous_update=True, 
                        on_value=set_input_ticker
                    )
                    sl.IconButton(
                        color="primary",
                        icon_name='mdi-chevron-right',
                        on_click=update_ticker,
                        disabled=input_ticker == "" or input_ticker == ticker
                    )
            else:
                FunAnimation()
        sl.HTML("div")

    if ticker is None or ticker == "":
        return
    
    if is_loading:
        Loader()

    info = None if data is None else data["info"]
    # Title
    if info is not None:
        sl.HTML("h1", f"{info['companyName']} ({info['symbol']})")

    # Percentage change across time periods
    PercentageChange(data)

    # Overview + price side by side
    with sl.Columns([1, 4], style="align-items: center;"):
        Overview(info)
        Price(data)

    # Description + AI-analysis side by side
    with sl.Columns([2, 3]):
        Description(info)
        AIAnalysis(ticker, data)

    # Charts for various data
    Charts(data)