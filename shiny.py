from shiny import App, ui, render, reactive
import pandas as pd
import matplotlib.pyplot as plt
import torch
import joblib
from a6_ex4 import PM_Model

pollutants = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_file("file", "Upload air_quality_cleaned.csv"),
        ui.input_selectize("pollutant", "Select pollutant(s)", pollutants, selected="PM2.5", multiple=True),
        ui.input_slider("window", "Rolling average window (days)", 1, 30, 7),
        ui.input_checkbox("show_pred", "Show PM2.5 prediction"),
        ui.input_file("scaler", "Upload scaler.save", accept=[".save"], multiple=False),
        ui.input_file("model", "Upload model.pt", accept=[".pt"], multiple=False),
    ),
    ui.output_plot("pollutant_plot")
)

def server(input, output, session):
    @reactive.Calc
    def data():
        file = input.file()
        if file is None:
            return None
        df = pd.read_csv(file[0]["datapath"])
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date")
        elif "datetime" in df.columns:
            df["Date"] = pd.to_datetime(df["datetime"])
            df = df.sort_values("Date")
        return df

    @output
    @render.plot
    def pollutant_plot():
        df = data()
        if df is None:
            return
        window = input.window()
        selected = input.pollutant()
        plt.figure(figsize=(10, 5))
        for pol in selected:
            if pol in df.columns:
                plt.plot(df["Date"], df[pol].rolling(window).mean(), label=pol)
        # PM2.5 prediction
        if input.show_pred() and input.scaler() and input.model():
            # Prepare features (drop target and non-numeric columns)
            features = df.select_dtypes("number").drop(columns=["PM2.5"], errors="ignore")
            scaler = joblib.load(input.scaler()[0]["datapath"])
            X = scaler.transform(features.values)
            input_dim = X.shape[1]
            model = PM_Model(input_dim)
            model.load_state_dict(torch.load(input.model()[0]["datapath"], map_location="cpu"))
            model.eval()
            with torch.no_grad():
                pred = model(torch.tensor(X, dtype=torch.float32)).numpy().flatten()
            plt.plot(df["Date"], pd.Series(pred).rolling(window).mean(), label="PM2.5 Prediction", linestyle="--")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.title("Pollutant Trends")
        plt.legend()
        plt.tight_layout()

app = App(app_ui, server)