# WC Predictor

Dự đoán kết quả bóng đá World Cup: 1X2, tỷ số chính xác, và xác suất vô địch,
dùng ensemble của Elo rating + Dixon-Coles (Poisson) + Gradient Boosting,
kết hợp Monte Carlo simulation cho toàn giải đấu.

## Kiến trúc

```
wc-predictor/
├── data/
│   ├── raw/                # Dataset gốc (tải từ Kaggle, không commit lên git)
│   └── processed/          # Dataset đã xử lý (features, elo history...)
├── src/
│   ├── data/
│   │   └── load_data.py    # Load + clean dữ liệu lịch sử các trận
│   ├── features/
│   │   └── build_features.py  # Tạo feature: elo_diff, form, h2h, ranking...
│   ├── models/
│   │   ├── elo.py           # Elo rating model (tự cập nhật theo thời gian)
│   │   ├── dixon_coles.py    # Poisson model có điều chỉnh cho tỷ số thấp
│   │   ├── gbm_model.py       # XGBoost/LightGBM classifier cho 1X2
│   │   └── ensemble.py        # Kết hợp 3 model trên (weighted / stacking)
│   └── simulation/
│       ├── bracket.py          # Cấu trúc bracket WC 2026 (48 đội, 12 bảng)
│       └── tournament_simulator.py  # Monte Carlo simulate cả giải
├── tests/
│   └── test_elo.py
├── notebooks/
│   └── 01_eda.ipynb          # (tự thêm) EDA trên dữ liệu lịch sử
├── requirements.txt
└── README.md
```

## Nguồn dữ liệu đề xuất

1. **Kaggle - International football results from 1872 to 2026** (martj42)
   https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017
   → Lịch sử toàn bộ các trận đấu quốc tế, có cập nhật đến gần đây.
   Tải file `results.csv` bỏ vào `data/raw/`.

2. **FIFA World Ranking** (Kaggle hoặc scrape fifa.com)
   → Dùng làm feature bổ sung ngoài Elo tự tính.

3. **(Tùy chọn) API-Football** (https://www.api-football.com/) free tier
   → Lấy lịch thi đấu WC 2026 thực tế + kết quả cập nhật real-time để
   test model trên các trận đang diễn ra.

> Elo tự tính trong `src/models/elo.py` từ lịch sử trận đấu, không cần
> phụ thuộc eloratings.net, nên chỉ cần dataset #1 là đủ chạy end-to-end.

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Tải data/raw/results.csv từ Kaggle (xem link trên) rồi chạy:
python -m src.data.load_data
python -m src.features.build_features
```

## Chạy pipeline

```bash
# 1. Train Elo history (chạy nhanh, thuần thuật toán)
python -m src.models.elo

# 2. Fit Dixon-Coles trên dữ liệu lịch sử
python -m src.models.dixon_coles

# 3. Train GBM classifier
python -m src.models.gbm_model

# 4. Ensemble + simulate cả giải đấu (Monte Carlo)
python -m src.simulation.tournament_simulator
```

## Roadmap

- [ ] Load + clean dữ liệu lịch sử
- [ ] Elo rating baseline
- [ ] Dixon-Coles cho tỷ số chính xác
- [ ] Feature engineering (form, h2h, ranking, home advantage)
- [ ] GBM classifier
- [ ] Ensemble (weighted → thử stacking sau)
- [ ] Bracket WC 2026 (48 đội) + Monte Carlo simulation
- [ ] (Optional) API FastAPI serve prediction + dashboard đơn giản
- [ ] Backtest trên các kỳ WC trước để đánh giá độ chính xác

## Đánh giá model

Backtest trên WC 2018 và WC 2022 (dùng dữ liệu trước giải để train, dự đoán
kết quả thật) bằng các metric:
- Log loss / Brier score cho xác suất 1X2
- RPS (Ranked Probability Score) — chuẩn phổ biến trong betting/football analytics
- Accuracy đơn giản để dễ hiểu

## License

MIT
