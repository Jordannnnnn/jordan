import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import mysql.connector
from flask import (
    Flask,
    render_template,
    redirect,
    url_for,
    request,
    make_response,
    flash,
    session,
    jsonify,
)

from datetime import date, timedelta, datetime
from pandas_datareader import data as pdr

import sys

import warnings

warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

plt.style.use("fivethirtyeight")
import mplfinance as mpf
from pylab import rcParams
from flask_cors import CORS

rcParams["figure.figsize"] = 10, 6
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima
from pandas_datareader.data import DataReader

# pip install setuptools
# pip install pandas_datareader
# pip install mysql-connector-python

app = Flask(__name__)
app.secret_key = b"fyp_24_s2_39"
CORS(app, resources={r"/*": {"origins": "*"}})

yf.pdr_override()

Start = date.today() - timedelta(365)
Start.strftime("%Y-%m-%d")
End = date.today() + timedelta(2)
End.strftime("%Y-%m-%d")


@app.route("/", methods=["POST", "GET"])
def index():
    error = None
    plot = None
    if request.method == "POST":
        ticker = request.form["stockTicker"]
        days = int(request.form["stockDays"])
        StartDay = date.today() - timedelta(days)
        StartDay.strftime("%Y-%m-%d")
        EndDay = date.today() + timedelta(2)
        EndDay.strftime("%Y-%m-%d")
        
        result = yf.download(tickers=ticker, start=StartDay, end=EndDay)['Adj Close']
        if len(result) > 0:
            stock = yf.Ticker(ticker=ticker)
            companyName = stock.info['longName']
            fig = plt.figure()
            plt.plot(result, linewidth=2)
            plt.title(f'{companyName} ({ticker})')
            plt.ylabel('Price ($)')
            plt.xlabel('Date')
            fig.autofmt_xdate()
            plt.tight_layout()
            plot=fig.savefig('static/images/guest_graph.png', dpi=fig.dpi, bbox_inches="tight")
            return render_template("guestSearchStock.html", plot="Success")
        else:
            error = (
                "No such stock ticker exists. Please try again! (Example: AAPL, GOOG)"
            )
            return render_template("index.html", error=error)
    return render_template("index.html")

@app.route("/about", methods=["POST", "GET"])
def about():
    return render_template("about.html")

@app.route("/stock/AAPL")
def apple():
    return render_template("register.html")

@app.route("/register", methods=["POST", "GET"])
def register():
    error = None
    success = None
    if request.method == "POST":
        username = request.form["username"]
        fname = request.form["fname"]
        lname = request.form["lname"]
        password = request.form["password"]
        email = request.form["email"]

        db = mysql.connector.connect(
            host = "db-mysql-sgp1-12968-do-user-17367918-0.j.db.ondigitalocean.com",
            user = "doadmin",
            password = "AVNS_ItKG7fksQ2ww_rQ7MLX",
            database = "market_prophet",
            port = 25060
        )

        cursor = db.cursor(buffered=True)
        cursor2 = db.cursor(buffered=True)
        checkDuplicateUsername = (
            """SELECT Username FROM Users WHERE Username = '%s'""" % (username)
        )
        checkDuplicateEmail = """SELECT Email FROM Users WHERE Email='%s'""" % (email)
        cursor.execute(checkDuplicateUsername)
        cursor2.execute(checkDuplicateEmail)
        result = cursor.fetchall()
        result2 = cursor2.fetchall()

        if len(result) > 0:
            error = "Username already exists. Please try again!"
        elif len(result2) > 0:
            error = "Email already exists. Please try again!"
        else:
            registerCredentials = "INSERT INTO Users (Username, Password, Email, FirstName, LastName) VALUES (%s, %s, %s, %s, %s)"
            values = (username, password, email, fname, lname)
            cursor.execute(registerCredentials, values)
            db.commit()
            success = "Welcome, " + str(username)
            session["username"] = str(username)
            # return render_template('userDashboard.html', success=success)
            return redirect(url_for("userDashboard", success=success))
    return render_template("register.html", error=error)


@app.route("/login", methods=["POST", "GET"])
def login():
    error = None
    success = None
    userWatchlist = []
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        db = mysql.connector.connect(
            host = "db-mysql-sgp1-12968-do-user-17367918-0.j.db.ondigitalocean.com",
            user = "doadmin",
            password = "AVNS_ItKG7fksQ2ww_rQ7MLX",
            database = "market_prophet",
            port = 25060
        )

        cursor = db.cursor(buffered=True)
        checkCredentials = (
            """SELECT Username, Password FROM Users WHERE Username = '%s' AND Password= '%s'"""
            % (username, password)
        )
        cursor.execute(checkCredentials)
        result = cursor.fetchall()

        if len(result) < 1:
            error = "Invalid credentials, please try again!"
        else:
            checkRoleType = """SELECT Role FROM Users WHERE Username = '%s'""" % (
                username
            )
            cursor.execute(checkRoleType)
            roleType = cursor.fetchall()
            for x in roleType:
                if str(x[0]) == "User":
                    cursor2 = db.cursor(buffered=True)
                    getUserID = """SELECT UserID from Users WHERE Username = '%s'""" % (
                        username
                    )
                    cursor2.execute(getUserID)
                    userID = cursor2.fetchall()
                    success = "Welcome, " + str(username)
                    session["username"] = str(username)
                    for y in userID:
                        cursor3 = db.cursor(buffered=True)
                        getWatchlist = (
                            """SELECT StockTicker from Watchlist WHERE UserID = '%s'"""
                            % (str(y[0]).replace("()", ""))
                        )
                        cursor3.execute(getWatchlist)
                        watchlist = cursor3.fetchall()
                        for z in watchlist:
                            userWatchlist.append(str(z[0]))
                    return redirect(url_for("userDashboard"))

                else:
                    success = "Welcome, " + str(username)
                    session["username"] = str(username)
                    return render_template("adminDashboard.html", success=success)
    return render_template("login.html", error=error)


@app.route("/userDashboard", methods=["POST", "GET"])
def userDashboard():
    error = None
    success = None
    userWatchlist = []

    if "username" in session:
        username = session["username"]

        if request.method == "POST":
            if request.form["button_identifier"] == "search_button":
                ticker = request.form["stockTicker"]
                session["ticker"] = ticker
                result = yf.download(tickers=ticker, start=Start, end=End)

                if len(result) > 0:
                    fullStockName = (
                        yf.Ticker(str(ticker)).info["shortName"] + f" ({ticker})"
                    )
                    return redirect(url_for("stockResult"))
                else:
                    error = "No such stock ticker exists. Please try again! (Example: AAPL, GOOG)"
                    return render_template("userDashboard.html", error=error)

        return render_template("userDashboard.html", userWatchlist=userWatchlist)
    else:
        error = "You are not logged in. Please log in first!"
        return render_template("login.html", error=error)


@app.route("/stockResult")
def stockResult():
    ticker = session["ticker"]
    result = yf.download(tickers=ticker, start=Start, end=End)
    fullStockName = yf.Ticker(str(ticker)).info["shortName"] + f" ({ticker})"
    stockTicker = str(ticker)
    return render_template(
        "stockResult.html", success=fullStockName, stockTicker=stockTicker
    )


@app.route("/stockDisplay", methods=["POST", "GET"])
def stockDisplay():
    # if request.method == 'GET':
    # return jsonify({"message": "GET request"})
    if request.method == "POST":
        data = request.get_json()
        stock_code = data["stock_code"]
        forecast_days = data["forecast_days"]
        print("Downloading stock data for {}...".format(stock_code))
        print(stock_code, forecast_days)
        result = yf.download(tickers=stock_code, start=Start, end=End)

        if len(result) > 0:
            try:
                stock_data = yf.download(
                    stock_code,
                    start="2022-01-01",
                    end=datetime.now().strftime("%Y-%m-%d"),
                )
                print("Download completed!")
                forecast_using_arima(stock_data, forecast_days, no_stats_log=True)
                generate_candleStick_plot_for_recommandations(stock_data)
                return jsonify({"message": "Chart Generated successfully!"})
            except Exception as e:
                print(f"Error fetching data: {e}")
                return jsonify({"message": "Chart Generation Failed!"})
            return render_template("stockResult.html")
        return render_template("stockResult.html")
    return render_template("stockResult.html")


def plot_seasonal_decomposition(df, model="multiplicative"):
    result = seasonal_decompose(df, model=model, period=30)
    fig = plt.figure()
    fig = result.plot()
    fig.set_size_inches(16, 9)


def get_best_arima_model_params(df, no_stats_log=False):
    trace = not no_stats_log
    model_autoARIMA = auto_arima(
        df,
        start_p=0,
        start_q=0,
        test="adf",  # use adftest to find optimal 'd'
        max_p=5,
        max_q=5,  # maximum p and q
        m=1,  # frequency of series
        d=None,  # let model determine 'd'
        seasonal=False,  # No Seasonality
        start_P=0,
        D=0,
        trace=trace,
        error_action="ignore",
        suppress_warnings=True,
        stepwise=True,
    )
    if no_stats_log == False:
        print(model_autoARIMA.summary())
        model_autoARIMA.plot_diagnostics(figsize=(15, 8))

    return model_autoARIMA.order


def plot_forecast(fitted_model, train_data, steps):
    fc = fitted_model.forecast(steps=steps)

    # If you need confidence intervals, use `get_forecast` instead
    forecast_object = fitted_model.get_forecast(steps=steps)
    se = forecast_object.se_mean
    conf = forecast_object.conf_int(alpha=0.55)

    fc = np.exp(fc)
    conf = np.exp(conf)
    train_data = np.exp(train_data)

    forecast_index = pd.date_range(
        start=train_data.index[-1] + pd.DateOffset(days=1), periods=steps, freq="D"
    )

    fc.index = forecast_index
    lower_series = pd.Series(conf.iloc[:, 0].values, index=forecast_index[: len(conf)])
    upper_series = pd.Series(conf.iloc[:, 1].values, index=forecast_index[: len(conf)])

    # Plotting
    plt.figure(figsize=(10, 5), dpi=100)
    plt.plot(train_data, label="training")
    plt.plot(fc, label="forecast")
    plt.fill_between(
        lower_series.index, lower_series, upper_series, color="k", alpha=0.15
    )
    plt.title("Projection of Stock Prices")
    plt.legend(loc="upper left", fontsize=8)
    plt.savefig("static/images/forecast_plot.png")


def forecast_using_arima(df, days_to_forecast, no_stats_log=False):
    steps = days_to_forecast

    df_close = df["Close"]

    if no_stats_log == False:
        plot_seasonal_decomposition(df_close)

    df_log = np.log(df_close)

    p, d, q = get_best_arima_model_params(df_log, no_stats_log)
    model = ARIMA(df_log, order=(p, d, q))
    fitted = model.fit()

    if no_stats_log == False:
        print(fitted.summary())

    plot_forecast(fitted, df_log, steps)


def generate_candleStick_plot_for_recommandations(stock_data):
    import time

    mpf.plot(
        stock_data.tail(100),
        type="candle",
        style="yahoo",
        ylabel="Price",
        savefig="static/images/trend_analysis_plot.png",
    )


@app.route("/watchlistView")
def watchlistView():
    userWatchlist = []

    if "username" in session:
        username = session["username"]

        db = mysql.connector.connect(
            host = "db-mysql-sgp1-12968-do-user-17367918-0.j.db.ondigitalocean.com",
            user = "doadmin",
            password = "AVNS_ItKG7fksQ2ww_rQ7MLX",
            database = "market_prophet",
            port = 25060
        )

        cursor = db.cursor(buffered=True)
        getUserID = """SELECT UserID from Users WHERE Username = '%s'""" % (username)
        cursor.execute(getUserID)
        userID = cursor.fetchall()

        for y in userID:
            cursor2 = db.cursor(buffered=True)
            getWatchlist = (
                """SELECT StockTicker from Watchlist WHERE UserID = '%s'"""
                % (str(y[0]).replace("()", ""))
            )
            cursor2.execute(getWatchlist)
            watchlist = cursor2.fetchall()
            for z in watchlist:
                userWatchlist.append(str(z[0]))
        return render_template("userWatchlist.html", userWatchlist=userWatchlist)

@app.route("/watchlistAdd")
def watchlistAdd():
    error = None
    success = None
    userWatchlist = []
    if "ticker" in session:
        ticker = session["ticker"]
        username = session["username"]

        db = mysql.connector.connect(
            host = "db-mysql-sgp1-12968-do-user-17367918-0.j.db.ondigitalocean.com",
            user = "doadmin",
            password = "AVNS_ItKG7fksQ2ww_rQ7MLX",
            database = "market_prophet",
            port = 25060
        )

        cursor = db.cursor(buffered=True)

        getUserID = "SELECT UserID from Users WHERE Username='%s'" % (username)
        cursor.execute(getUserID)
        result = cursor.fetchall()

        if len(result) > 0:
            for x in result:
                cursor2 = db.cursor(buffered=True)
                checkDuplicate = (
                    "SELECT StockTicker from Watchlist WHERE StockTicker='%s' and UserID = '%s'"
                    % (ticker, str(x[0]))
                )
                cursor2.execute(checkDuplicate)
                checkDuplicateResult = cursor2.fetchall()
                if len(checkDuplicateResult) < 1:
                    cursor3 = db.cursor(buffered=True)
                    updateWatchlist = (
                        "INSERT INTO Watchlist (UserID, StockTicker) VALUES (%s, %s)"
                    )
                    values = (str(x[0]), ticker)
                    cursor3.execute(updateWatchlist, values)
                    db.commit()
                    success = "Successfully added in your watchlist!"

                    cursor4 = db.cursor(buffered=True)
                    getWatchlist = (
                        """SELECT StockTicker from Watchlist WHERE UserID = '%s'"""
                        % (str(x[0]))
                    )
                    cursor4.execute(getWatchlist)
                    watchlist = cursor4.fetchall()
                    for z in watchlist:
                        userWatchlist.append(str(z[0]))
                    return redirect(url_for("userDashboard"))
                else:
                    error = "You already have this stock in your watchlist!"
                    return render_template("stockResult.html", error=error)

    return render_template("stockResult.html", error=error)


@app.route("/watchlistAction", methods=["POST", "GET"])
def watchlistRemove():
    if request.method == "POST":
        if request.form["button_identifier"] == "watchlist_button":
            checkRemoveBtn = request.form.get("removeStock")
            checkDisplayBtn = request.form.get("displayStock")
            checkDaysInput = request.form.get("numOfDays")

            if checkRemoveBtn is not None:
                username = session["username"]
                stockToDelete = request.form["removeStock"]

                db = mysql.connector.connect(
                    host = "db-mysql-sgp1-12968-do-user-17367918-0.j.db.ondigitalocean.com",
                    user = "doadmin",
                    password = "AVNS_ItKG7fksQ2ww_rQ7MLX",
                    database = "market_prophet",
                    port = 25060
                )

                cursor = db.cursor(buffered=True)
                getUserID = "SELECT UserID from Users WHERE Username = '%s'" % (
                    username
                )
                cursor.execute(getUserID)
                result = cursor.fetchall()
                for x in result:
                    cursor2 = db.cursor(buffered=True)
                    deleteStock = (
                        """DELETE FROM Watchlist WHERE UserID = '%s' and StockTicker = '%s'"""
                        % (x[0], str(stockToDelete))
                    )
                    cursor2.execute(deleteStock)
                    db.commit()
                return redirect(url_for("userDashboard"))

            elif checkDisplayBtn is not None:
                ticker = request.form["displayStock"]
                checkEmptyInput = ""
                if checkDaysInput == checkEmptyInput or checkDaysInput is None:
                    result = yf.download(tickers=ticker, start=Start, end=End)['Adj Close']
                    if len(result) > 0:
                        stock = yf.Ticker(ticker=ticker)
                        companyName = stock.info['longName']
                        fig = plt.figure()
                        plt.plot(result, linewidth=2)
                        plt.title(f'{companyName} ({ticker})')
                        plt.ylabel('Price ($)')
                        plt.xlabel('Date')
                        fig.autofmt_xdate()
                        plt.tight_layout()
                        plot=fig.savefig('static/images/user_graph.png', dpi=fig.dpi, bbox_inches="tight")
                        return render_template("userSearchStock.html", plot="Success")
                else:
                    days = int(request.form["numOfDays"])
                    StartDay = date.today() - timedelta(days)
                    StartDay.strftime("%Y-%m-%d")
                    EndDay = date.today() + timedelta(2)
                    EndDay.strftime("%Y-%m-%d")
                    result = yf.download(tickers=ticker, start=StartDay, end=EndDay)['Adj Close']
                    if len(result) > 0:
                        stock = yf.Ticker(ticker=ticker)
                        companyName = stock.info['longName']
                        fig = plt.figure()
                        plt.plot(result, linewidth=2)
                        plt.title(f'{companyName} ({ticker})')
                        plt.ylabel('Price ($)')
                        plt.xlabel('Date')
                        fig.autofmt_xdate()
                        plt.tight_layout()
                        plot=fig.savefig('static/images/user_graph.png', dpi=fig.dpi, bbox_inches="tight")
                        return render_template("userSearchStock.html", plot="Success")
        return render_template("userDashboard.html")

@app.route("/viewProfile", methods=["POST", "GET"])
def viewProfile():
    if "username" in session:
        details = []
        username = session["username"]

        db = mysql.connector.connect(
            host = "db-mysql-sgp1-12968-do-user-17367918-0.j.db.ondigitalocean.com",
            user = "doadmin",
            password = "AVNS_ItKG7fksQ2ww_rQ7MLX",
            database = "market_prophet",
            port = 25060
        )

        cursor = db.cursor()
        displayUserDetails = (
            """SELECT Username, Email, FirstName, LastName from Users WHERE Username = '%s'"""
            % (username)
        )
        cursor.execute(displayUserDetails)
        result = cursor.fetchall()
        for row in result:
            details.append(row)
        return render_template("userViewProfile.html", details=details)


@app.route("/updateProfile", methods=["POST", "GET"])
def updateProfile():
    error = None
    success = None
    if request.method == "POST":
        username = request.form["username"]
        current_password = request.form["current_password"]
        new_password = request.form["new_password"]
        fname = request.form["fname"]
        lname = request.form["lname"]

        db = mysql.connector.connect(
            host = "db-mysql-sgp1-12968-do-user-17367918-0.j.db.ondigitalocean.com",
            user = "doadmin",
            password = "AVNS_ItKG7fksQ2ww_rQ7MLX",
            database = "market_prophet",
            port = 25060
        )

        cursor = db.cursor(buffered=True)
        checkExistingUser = (
            """SELECT Username, Password FROM Users WHERE Username='%s' and Password='%s'"""
            % (username, current_password)
        )
        cursor.execute(checkExistingUser)
        result = cursor.fetchall()

        if len(result) < 1:
            error = "Invalid credentials, please try again!"
        else:
            updateCredentials = """UPDATE Users SET Password = %s, firstName=%s, lastName=%s WHERE Username = %s"""
            values = (new_password, fname, lname, username)
            cursor.execute(updateCredentials, values)
            db.commit()
            success = "Profile successfully updated!"
            return render_template("userUpdateParticular.html", success=success)

    return render_template("userUpdateParticular.html", error=error)


@app.route("/deleteProfile", methods=["POST", "GET"])
def deleteProfile():
    if request.method == "POST":
        if "username" in session:
            username = session["username"]

            db = mysql.connector.connect(
            host = "db-mysql-sgp1-12968-do-user-17367918-0.j.db.ondigitalocean.com",
            user = "doadmin",
            password = "AVNS_ItKG7fksQ2ww_rQ7MLX",
            database = "market_prophet",
            port = 25060
        )

            cursor = db.cursor()
            getUserID = "SELECT UserID from Users where Username = '%s'" % (username)
            cursor.execute(getUserID)
            result = cursor.fetchall()
            for x in result:
                cursor = db.cursor()
                deleteUser = "DELETE FROM Users WHERE UserID = '%s'" % (x[0])
                cursor.execute(deleteUser)
                db.commit()
            return redirect(url_for("index"))
            session.pop("username")

@app.route('/submitFeedback', methods=["POST","GET"])
def submitFeedback():
    if request.method == "POST":
        title = request.form['feedbackTitle']
        content = request.form['feedbackContent']

        if not title:
            flash('Title is required!')
        elif not content:
            flash('Content is required!')
        else:
            db = mysql.connector.connect(
            host = "db-mysql-sgp1-12968-do-user-17367918-0.j.db.ondigitalocean.com",
            user = "doadmin",
            password = "AVNS_ItKG7fksQ2ww_rQ7MLX",
            database = "market_prophet",
            port = 25060
            )
            cursor = db.cursor()
            insertUserFeedBack = "INSERT INTO Feedback(Title, Content) VALUES (%s, %s)"
            values = (title, content)
            cursor.execute(insertUserFeedBack, values)
            db.commit()
            return redirect(url_for('userDashboard'))

    else:
        return render_template('userSubmitFeedback.html')

@app.route('/viewFeedback')
def viewFeedback():
    feedback = []

    db = mysql.connector.connect(
            host = "db-mysql-sgp1-12968-do-user-17367918-0.j.db.ondigitalocean.com",
            user = "doadmin",
            password = "AVNS_ItKG7fksQ2ww_rQ7MLX",
            database = "market_prophet",
            port = 25060
            )
    cursor = db.cursor()
    getFeedback = "SELECT FeedbackID, Title, Content FROM Feedback"
    cursor.execute(getFeedback)
    result = cursor.fetchall()
    for x in result:
        feedback.append(x)
    return render_template('adminViewFeedback.html', feedback=feedback)

@app.route('/deleteFeedback', methods=["POST","GET"])
def deleteFeedback():
    if request.method == "POST":
        feedbackID = request.form.get("deleteFeedback")

        db = mysql.connector.connect(
            host = "db-mysql-sgp1-12968-do-user-17367918-0.j.db.ondigitalocean.com",
            user = "doadmin",
            password = "AVNS_ItKG7fksQ2ww_rQ7MLX",
            database = "market_prophet",
            port = 25060
            )

        cursor = db.cursor()
        deleteFeedback = "DELETE FROM Feedback WHERE FeedbackID = %s" %(feedbackID)
        cursor.execute(deleteFeedback)
        db.commit()
        return render_template('adminDashboard.html')


@app.route("/adminDashboard")
def adminDashboard():
    if "username" in session:
        return render_template("adminDashboard.html")
    else:
        error = "You are not logged in. Please try again!"
        return render_template("login.html", error=error)


@app.route("/adminSearch", methods=["POST", "GET"])
def adminSearch():
    error = None
    details = []
    if request.method == "POST":
        username = request.form["username"]

        db = mysql.connector.connect(
            host = "db-mysql-sgp1-12968-do-user-17367918-0.j.db.ondigitalocean.com",
            user = "doadmin",
            password = "AVNS_ItKG7fksQ2ww_rQ7MLX",
            database = "market_prophet",
            port = 25060
        )

        cursor = db.cursor(buffered=True)
        checkExistingUser = (
            """SELECT Username from Users WHERE Username='%s' AND NOT Role = 'Admin'"""
            % (username)
        )
        cursor.execute(checkExistingUser)
        result = cursor.fetchall()

        if len(result) < 1:
            error = "User does not exist. Please try again!"
        else:
            print(result)
            displayUserDetails = (
                """SELECT Username, Email, FirstName, LastName from Users WHERE Username='%s'"""
                % (username)
            )
            cursor.execute(displayUserDetails)
            result = cursor.fetchall()
            for row in result:
                details.append(row)
            return render_template("adminSearchUser.html", details=details)
    return render_template("adminDashboard.html", error=error)


@app.route("/adminUpdate", methods=["POST", "GET"])
def adminUpdate():
    if request.method == "POST":
        if request.form["button_identifier"] == "admin_button":
            checkEditBtn = request.form.get("editUserDetails")
            checkBackBtn = request.form.get("back")
            checkUpdateBtn = request.form.get("updateUserDetails")
            checkDeleteBtn = request.form.get("deleteUser")

            if checkDeleteBtn is not None:
                username = request.form["username"]
                db = mysql.connector.connect(
                    host = "db-mysql-sgp1-12968-do-user-17367918-0.j.db.ondigitalocean.com",
                    user = "doadmin",
                    password = "AVNS_ItKG7fksQ2ww_rQ7MLX",
                    database = "market_prophet",
                    port = 25060
                )
                cursor = db.cursor()
                getUserID = """SELECT UserID from Users WHERE Username = '%s'""" % (
                    username
                )
                cursor.execute(getUserID)
                result = cursor.fetchall()
                for x in result:
                    cursor = db.cursor()
                    deleteUser = "DELETE FROM Users WHERE UserID = '%s'" % (x[0])
                    cursor.execute(deleteUser)
                    db.commit()
                flash("User successfully deleted!")

            elif checkEditBtn is not None:
                userDetails = []
                username = request.form["username"]
                db = mysql.connector.connect(
                    host = "db-mysql-sgp1-12968-do-user-17367918-0.j.db.ondigitalocean.com",
                    user = "doadmin",
                    password = "AVNS_ItKG7fksQ2ww_rQ7MLX",
                    database = "market_prophet",
                    port = 25060
                )
                cursor = db.cursor()
                getUserDetails = (
                    """SELECT Username, Email, FirstName, LastName FROM Users WHERE Username='%s'"""
                    % (username)
                )
                cursor.execute(getUserDetails)
                result = cursor.fetchall()
                for row in result:
                    userDetails.append(row)
                return render_template(
                    "adminUpdateUserParticular.html", userDetails=userDetails
                )
            elif checkBackBtn is not None:
                return redirect(url_for("adminDashboard"))
            elif checkUpdateBtn is not None:
                username = request.form["username"]
                email = request.form["email"]
                firstName = request.form["fname"]
                lastName = request.form["lname"]
                db = mysql.connector.connect(
                    host = "db-mysql-sgp1-12968-do-user-17367918-0.j.db.ondigitalocean.com",
                    user = "doadmin",
                    password = "AVNS_ItKG7fksQ2ww_rQ7MLX",
                    database = "market_prophet",
                    port = 25060
                )
                if not (email):
                    cursor = db.cursor()
                    updateDetails = """UPDATE Users set FirstName = %s, LastName = %s WHERE Username = %s"""
                    values = (firstName, lastName, username)
                    cursor.execute(updateDetails, values)
                    db.commit()
                    print("case 1")
                elif not (firstName):
                    cursor = db.cursor()
                    updateDetails = """UPDATE Users set Email= %s, LastName = %s WHERE Username = %s"""
                    values = (email, lastName, username)
                    cursor.execute(updateDetails, values)
                    db.commit()
                    print("case 2")
                elif not (lastName):
                    cursor = db.cursor()
                    updateDetails = """UPDATE Users set Email= %s, FirstName = %s WHERE Username = %s"""
                    values = (email, firstName, username)
                    cursor.execute(updateDetails, values)
                    db.commit()
                    print("case 3")
                elif not (email, firstName):
                    cursor = db.cursor()
                    updateDetails = (
                        """UPDATE Users set LastName= %s WHERE Username = %s"""
                    )
                    values = (lastName, username)
                    cursor.execute(updateDetails, values)
                    db.commit()
                    print("case 4")
                elif not (email, lastName):
                    cursor = db.cursor()
                    updateDetails = (
                        """UPDATE Users set FirstName = %s WHERE Username = %s"""
                    )
                    values = (firstName, username)
                    cursor.execute(updateDetails, values)
                    db.commit()
                    print("case 5")
                elif not (firstName, lastName):
                    cursor = db.cursor()
                    updateDetails = """UPDATE Users set Email= %s WHERE Username = %s"""
                    values = (email, username)
                    cursor.execute(updateDetails, values)
                    db.commit()
                    print("case 6")
                else:
                    cursor = db.cursor()
                    updateDetails = """UPDATE Users set Email = %s, FirstName = %s, LastName = %s WHERE Username = %s"""
                    values = (email, firstName, lastName, username)
                    cursor.execute(updateDetails, values)
                    db.commit()
                    print("case 7")
        return redirect(url_for("adminDashboard"))
    return render_template("adminDashboard.html", error=error)


@app.route("/logout")
def logout():
    session.pop("username")
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
