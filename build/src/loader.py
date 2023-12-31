import solara as sl
from typing import Optional
from solara.alias import rv

@sl.component
def Loader(text: Optional[str]=None):
    sl.Style("""
        .loader {
            font-size: 3em;
            animation: spinner 0.75s infinite;
            color: green !important;
        }
             
        @keyframes spinner {
            0% {
                transform: rotate(0deg);
            }
            100% {
                transform: rotate(359deg);
            }
        }
             
        .loader-text {
            font-size: 1.5em;
            font-weight: 500;
            margin-top: 1em;
        }
    """)

    with sl.Column(style="text-align: center;"):
        rv.Icon(children=['mdi-currency-usd'], class_="loader", style_="font-size: 3em;")
        if text is not None:
            sl.HTML("div", unsafe_innerHTML=text, class_="loader-text")