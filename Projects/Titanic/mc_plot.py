"""This Scripts collect all costermized plotly plot by coomike
"""
from typing import List, Optional
import pandas as pd
import plotly.express as px
from plotly.graph_objs import layout
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def mc_boxplots_solo(df: pd.DataFrame, col_names: List[str], units: List[str]) -> None:
    """Generate a row of box plots based on df and feature names
        """
    fig = make_subplots(rows=1, cols=len(col_names), subplot_titles=col_names)
    for i in range(len(col_names)):
        col_name = col_names[i]
        fig.add_trace(
            go.Box(y=df[col_name], name=units[i], showlegend=False),
            row=1, col=i + 1
        )
        fig.update_xaxes(title_text=col_name, row=1, col=i+1)
        fig.update_yaxes(title_text=units[i], row=1, col=i+1)
    fig.show()


def mc_boxplots_xy(df: pd.DataFrame, col_names: List[str], target: str, base: str) -> None:
    fig = make_subplots(rows=1, cols=len(col_names))
    for i in range(len(col_names)):
        col_name = col_names[i]
        px_fig = px.box(data_frame=df, x=base, y=col_name, color=target)
        for trace in px_fig['data']:
            if i != 0:
                trace['showlegend'] = False
            fig.add_trace(trace, row=1, col=i+1)

        fig.update_xaxes(title_text=base, row=1, col=i+1)
        fig.update_yaxes(title_text=col_names[i], row=1, col=i+1)
        fig.update_layout(boxmode='group', legend=px_fig['layout']['legend'])
    fig.show()


def mc_histgram(df: pd.DataFrame, col_names: List[str], target: str) -> None:
    """Generate a row of hist from df based on categorical columns and the target feature
        """
    # TODO how to set nbins with optional value?
    fig = make_subplots(rows=1, cols=len(col_names), subplot_titles=col_names)
    for i in range(len(col_names)):
        col_name = col_names[i]
        traces, layout = px_convert(px.histogram(
            data_frame=df, x=col_name, color=target))
        for trace in traces:
            if i != 0:
                trace['showlegend'] = False
            fig.add_trace(trace, row=1, col=i+1)
    fig.update_layout(barmode='stack', legend=layout['legend'])
    fig.show()


def px_convert(px_fig: px) -> List[any]:
    """convert px object to traces for later manipulation
        """
    try:
        data, layout = px_fig['data'], px_fig['layout']
        return (list(data), layout)
    except:
        print('something is wrong')


def mc_df_bar(df: pd.DataFrame, col_names: List[str], target: str, subgroup: str = None) -> None:
    """Given a pandas dataframe and categorical feature names and a target feature, draw error bars of each feature to target feature.
        """
    # TODO 1 how is a error band calculated in sns bar chart?
    fig = make_subplots(rows=1, cols=len(col_names))
    for i in range(len(col_names)):
        feature = col_names[i]
        heights = df[[feature, target]].groupby(feature, as_index=False).mean()
        # print(heights)
        errors = df[[feature, target]].groupby(feature, as_index=False).var()
        heights.columns = [feature, 'heights']
        errors.columns = [feature, 'errors']
        error_bar_df = pd.merge(heights, errors)
        # print(error_bar_df)
        # TODO with subgroup info
        trace, layout = mc_bar(error_bar_df, feature, target)

        trace['showlegend'] = False
        fig.add_trace(trace, row=1, col=i + 1)
        fig.update_xaxes(title_text=feature, row=1, col=i+1)
        if i == 0:
            fig.update_yaxes(title_text=target, row=1, col=i+1)
    fig.show()


def mc_bar(df: pd.DataFrame, feature: str, target: str):
    """Given calculated dataframe with feature, heights and errors columns, draw error bar"""
    names, heights, errors = [
        str(x) for x in df[feature]], df['heights'], df['errors']
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=names, y=heights,
        error_y=dict(type='data', array=errors)
    ))
    fig.update_xaxes(title_text=feature)
    fig.update_yaxes(title_text=target)
    trace = fig['data'][0]
    layout = fig['layout']
    # trace['showlegend'] = False
    # fig.show()
    # print(layout)
    return (trace, layout)


if __name__ == '__main__':
    print('main')
