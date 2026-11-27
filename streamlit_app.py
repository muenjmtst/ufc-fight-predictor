import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000" # это просто адрес FastApi

st.set_page_config(page_title='UFC Fight Prediction', layout='wide')# растягивает интерфейс на всю ширину экрана, чтобы две колонки бойцов не слипались.
st.title("UFC fight predictor")

with st.sidebar: #все рисуется в левой выдвижной панели
    st.header('Backend Status')
    try:
        #добавляю проверку health так как может выйти ошибка
        health_response = requests.get(f"{API_URL}/health", timeout=2)
        health_response.raise_for_status()
        health = health_response.json()

        #такая же тема с info
        info_response = requests.get(f"{API_URL}/model_info", timeout=2)
        info_response.raise_for_status()
        info = info_response.json()


        st.success(f"API: {health['status'].upper()}")
        st.write(f"Model name: {info['model_type']}")
        st.write(f"Model threshold: {info['threshold']}")
        st.write(f"Model features: {info['n_features']}")

    except requests.exceptions.RequestException:
        st.error("FastAPI is offline. Run uvicorn or Docker container first!")

st.subheader("Fight Details")
f_col1, f_col2, f_col3 = st.columns(3)
with f_col1:
    weight_class = st.selectbox(
        "Weight Class",
        ["Lightweight", "Welterweight", "Middleweight", "Light Heavyweight",
         "Heavyweight", "Featherweight", "Bantamweight", "Flyweight"]
    )
with f_col2:
    no_of_rounds = st.selectbox("Scheduled Rounds", [3, 5], index=0)#index=0, говорит то что по умолчанию будет выбрано 3 раунда количество
with f_col3:
    title_bout = int(st.checkbox("Title Bout", value=False))

st.divider()

col_red, col_blue = st.columns(2)

with col_red:
    st.subheader("🔴 Red Corner")
    r_age = st.number_input("Age (Red)", min_value=18, max_value=55, value=30)
    r_height = st.number_input("Height cm (Red)", min_value=140.0, max_value=220.0, value=180.0)
    r_reach = st.number_input("Reach cm (Red)", min_value=140.0, max_value=230.0, value=183.0)
    r_stance = st.selectbox("Stance (Red)", ["Orthodox", "Southpaw", "Switch", "Open Stance"])
    r_wins = st.number_input("Wins (Red)", min_value=0, max_value=60, value=15)
    r_losses = st.number_input("Losses (Red)", min_value=0, max_value=40, value=4)
    r_draws = st.number_input("Draws (Red)", min_value=0, max_value=10, value=0)
    r_win_streak = st.number_input("Current Win Streak (Red)", min_value=0, max_value=20, value=2)
    r_lose_streak = st.number_input("Current Lose Streak (Red)", min_value=0, max_value=15, value=0)
    r_rounds = st.number_input("Total Rounds Fought (Red)", min_value=0, max_value=150, value=25)
    r_title_bouts = st.number_input("Total Title Bouts (Red)", min_value=0, max_value=20, value=0)
    r_sig_str = st.slider("Sig. Strikes Accuracy % (Red)", 0.0, 1.0, 0.48)
    r_td_landed = st.number_input("Avg Takedowns Landed (Red)", min_value=0.0, max_value=15.0, value=1.5)
    r_td_pct = st.slider("Takedown Accuracy % (Red)", 0.0, 1.0, 0.40)
    r_sub_att = st.number_input("Avg Submission Attempts (Red)", min_value=0.0, max_value=10.0, value=0.6)
    r_ko = st.number_input("Wins by KO/TKO (Red)", min_value=0, max_value=40, value=7)
    r_sub = st.number_input("Wins by Submission (Red)", min_value=0, max_value=40, value=3)

with col_blue:
    st.subheader("🔵 Blue Corner")
    b_age = st.number_input("Age (Blue)", min_value=18, max_value=55, value=28)
    b_height = st.number_input("Height cm (Blue)", min_value=140.0, max_value=220.0, value=178.0)
    b_reach = st.number_input("Reach cm (Blue)", min_value=140.0, max_value=230.0, value=180.0)
    b_stance = st.selectbox("Stance (Blue)", ["Orthodox", "Southpaw", "Switch", "Open Stance"])
    b_wins = st.number_input("Wins (Blue)", min_value=0, max_value=60, value=12)
    b_losses = st.number_input("Losses (Blue)", min_value=0, max_value=40, value=5)
    b_draws = st.number_input("Draws (Blue)", min_value=0, max_value=10, value=0)
    b_win_streak = st.number_input("Current Win Streak (Blue)", min_value=0, max_value=20, value=1)
    b_lose_streak = st.number_input("Current Lose Streak (Blue)", min_value=0, max_value=15, value=0)
    b_rounds = st.number_input("Total Rounds Fought (Blue)", min_value=0, max_value=150, value=20)
    b_title_bouts = st.number_input("Total Title Bouts (Blue)", min_value=0, max_value=20, value=0)
    b_sig_str = st.slider("Sig. Strikes Accuracy % (Blue)", 0.0, 1.0, 0.44)
    b_td_landed = st.number_input("Avg Takedowns Landed (Blue)", min_value=0.0, max_value=15.0, value=1.0)
    b_td_pct = st.slider("Takedown Accuracy % (Blue)", 0.0, 1.0, 0.35)
    b_sub_att = st.number_input("Avg Submission Attempts (Blue)", min_value=0.0, max_value=10.0, value=0.4)
    b_ko = st.number_input("Wins by KO/TKO (Blue)", min_value=0, max_value=40, value=5)
    b_sub = st.number_input("Wins by Submission (Blue)", min_value=0, max_value=40, value=2)


st.divider()

#type=primary, чтобы визуально отделить от secondary buttons
#user_container_width=True, растягивает на 100 процентов ширины родительского элемента
if st.button("Predict Winner", type="primary", use_container_width=True): 
    r_exp = r_wins + r_losses + r_draws
    b_exp = b_wins + b_losses + b_draws

    r_wr = r_wins / r_exp if r_exp > 0 else None
    b_wr = b_wins / b_exp if b_exp > 0 else None

    if (r_wr is None) or (b_wr is None):
        win_rate_diff = None
    else:
        win_rate_diff = r_wr - b_wr

    payload = {
        "age_diff": r_age - b_age,
        "reach_diff": r_reach - b_reach,
        "height_diff": r_height - b_height,
        "win_streak_diff": r_win_streak - b_win_streak,
        "lose_streak_diff": r_lose_streak - b_lose_streak,
        "wins_diff": r_wins - b_wins,
        "losses_diff": r_losses - b_losses,
        "experience_diff": r_exp - b_exp,
        "win_rate_diff": win_rate_diff,
        "R_is_debutant": int(r_exp == 0),
        "B_is_debutant": int(b_exp == 0),
        "R_age": r_age,
        "R_Height_cms": r_height,
        "R_Reach_cms": r_reach,
        "R_wins": r_wins,
        "R_losses": r_losses,
        "R_current_win_streak": r_win_streak,
        "R_current_lose_streak": r_lose_streak,
        "R_total_rounds_fought": r_rounds,
        "R_total_title_bouts": r_title_bouts,
        "R_avg_SIG_STR_pct": r_sig_str,
        "R_avg_TD_landed": r_td_landed,
        "R_avg_TD_pct": r_td_pct,
        "R_avg_SUB_ATT": r_sub_att,
        "R_win_by_KO/TKO": r_ko,
        "R_win_by_Submission": r_sub,
        "R_Stance": r_stance,
        "B_age": b_age,
        "B_Height_cms": b_height,
        "B_Reach_cms": b_reach,
        "B_wins": b_wins,
        "B_losses": b_losses,
        "B_current_win_streak": b_win_streak,
        "B_current_lose_streak": b_lose_streak,
        "B_total_rounds_fought": b_rounds,
        "B_total_title_bouts": b_title_bouts,
        "B_avg_SIG_STR_pct": b_sig_str,
        "B_avg_TD_landed": b_td_landed,
        "B_avg_TD_pct": b_td_pct,
        "B_avg_SUB_ATT": b_sub_att,
        "B_win_by_KO/TKO": b_ko,
        "B_win_by_Submission": b_sub,
        "B_Stance": b_stance,
        "weight_class": weight_class,
        "title_bout": title_bout,
        "no_of_rounds": no_of_rounds,
    }

    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=5) #json=payload, превращает payload в json строку
        response.raise_for_status() # выводит http запрос с сервера, все ли норм или нет
        result = response.json()

        #добавляю новую защиту
        if not isinstance(result, dict):
            raise ValueError("API response must be a dictionary")

        #проверяю наличие нужных полей для предсказания, если их нет то все сломается
        required_fields = [
            "predicted_winner",
            "red_win_probability",
            "blue_win_probability",
            "threshold"
        ]

        for field in required_fields:
            if field not in result:
                raise ValueError(f"Missing field: {field}")



        winner = result["predicted_winner"]
        red_prob = result["red_win_probability"]
        blue_prob = result["blue_win_probability"]
        threshold = result["threshold"]

        if winner not in ("Red", "Blue"):
            raise ValueError("Invalid predicted winner")

        #проверка то что вероятность в промежутке от 0 до 1
        for value in (red_prob, blue_prob, threshold):
            if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0 <= value <= 1):
                raise ValueError("Invalid probability or threshold")


        #success - рисует зеленую плашку
        #info - рисует синию плашку
        if winner == "Red":
            st.success(f"🏆 Predicted Winner: **Red Corner** ({red_prob:.1%})")
        else:
            st.info(f"🏆 Predicted Winner: **Blue Corner** ({blue_prob:.1%})")

        m1, m2, m3 = st.columns(3)
        m1.metric("🔴 Red Win Probability", f"{red_prob:.1%}")
        m2.metric("🔵 Blue Win Probability", f"{blue_prob:.1%}")
        m3.metric("Decision Threshold", f"{threshold:.2f}")

        st.progress(red_prob)# рисует линию прогресса 
    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")

    except (ValueError, TypeError) as e:
        st.error(f"Invalid API response: {e}")


