import streamlit as st
import glob
import pandas as pd
import plotly.express as px
import numpy as np

st.write("""
<style>
.body { 
    font-family: 'Meiryo UI’;
    width: 800px;
}
</style>
""", unsafe_allow_html=True)

@st.cache
def read_csv():
    return pd.read_csv("input/commonlitreadabilityprize/train.csv")

@st.cache
def indent_excerpt(x):
    x = x.split(" ")
    ret = ""
    for i in range(len(x) // 5 + 1):
        ret += " ".join(ret[i*5:(i+1)*5]) + "\n"
    return ret

exp_names = [""] + glob.glob("output/*")
exp_name = st.selectbox(label="exp_name", options=exp_names)
fold = st.selectbox(label="fold", options=["", 0, 1, 2, 3, 4])

if exp_name != "":
    exp_files = [""] + glob.glob(f"{exp_name}/*")
    exp_file = st.selectbox(label="exp_file", options=exp_files)
    if exp_file != "":
        steps_dict = {i: x for i, x in enumerate(glob.glob(f"{exp_file}/val_fold{fold}*_step*.csv"))}
        steps = st.number_input(label="epoch", min_value=-1, max_value=len(steps_dict), step=1)
        df = read_csv()
        if steps == -1:
            df_oof = pd.concat([pd.read_csv(x) for x in glob.glob(f"{exp_file}/val_fold{fold}*_best.csv")])
        else:
            df_oof = pd.read_csv(steps_dict[steps])
        df = pd.merge(df, df_oof[["id", "pred"]], how="inner")
        df["mse"] = (df["pred"] - df["target"])**2
        rmse = np.sqrt(1 / len(df.values) * df["mse"].sum())
        st.write(f"rmse: {rmse}")
        df["hover_text"] = "id: " + df["id"].astype(str) + "<br>mse: " + df["mse"].round(4).astype(str)

        fig = px.scatter(data_frame=df,
                         x="pred",
                         y="target",
                         hover_name="hover_text",
                         range_x=(-4, 2),
                         range_y=(-4, 2),
                         height=500)
        st.plotly_chart(fig)

        fig = px.histogram(data_frame=df,
                           x="mse",
                           nbins=50,
                           range_x=(0, 4),
                           height=250)
        st.plotly_chart(fig)

        id_box = st.text_input(label="検索したいIDを入力してください")

        if id_box == "":
            st.table(df.sort_values("id")[["id", "excerpt", "target", "pred"]].head(5))
        else:
            st.table(df[df["id"] == id_box][["id", "excerpt", "target", "pred"]])

