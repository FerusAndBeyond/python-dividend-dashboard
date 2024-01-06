## Financial Dashboard for Dividend Stocks

This is a Python web app built with [Solara](https://github.com/widgetti/solara) where you can analyze dividend stocks. 

* [Article](https://medium.com/@dreamferus/creating-financial-dashboards-in-python-with-solara-70c82f39391d?sk=2552f997f26ae15528b4491c7c1bb6ba)
* [Video](https://www.youtube.com/watch?v=0DzHakZImvU)

## Setup

First clone the repository:

```bash
git clone git@github.com:FerusAndBeyond/python-dividend-dashboard.git
```
 
and then cd into it
 
```bash
cd python-dividend-dashboard
```

#### Environment variables

Run `init.sh`:

```bash
sh init.sh
```

This will create a `.env` file for configuration variables such as the needed API keys. The following APIs are used:

###### Financial Modeling Prep

Financial Modeling Prep (FMP) is a financial data API containing various data for fundamental analysis. You can sign up for free to obtain 250 API requests per day. Alternatively, for a premium version with more requests and additional features, you can sign up for a paid version. You can support me by using my affiliate link while also getting 15% off [here](https://utm.guru/uggRv). 

More info about FMP [here](https://site.financialmodelingprep.com/developer/docs).

###### OpenAI API

OpenAI, the company that created ChatGPT, has an API that you can use to analyze and generate text using AI. More info [here](https://platform.openai.com/docs/overview).

---

Add the two API keys to the `.env` file for variables `FMP_API_KEY` and `OPENAI_API_KEY`.

#### Run

There are two options, run in docker or outside of docker. To run in docker, you need to first have docker installed. Then, simply call:

###### Docker

```bash
sh run_docker.sh
```

Then open http://localhost:5000/.

To shut it down, use `sh down.sh`.

###### Outside of Docker

To run outside of Docker, first install the dependencies:

```bash
pip install -r build/requirements.txt
```

Then run the app:

```bash
sh run.sh
```

and open http://localhost:8765/.