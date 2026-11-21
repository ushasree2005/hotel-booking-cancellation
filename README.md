# Hotel Booking Cancellation Prediction

Predicting which hotel bookings will be cancelled, so a revenue team can decide where to ask for a deposit, send a reminder, or safely overbook.

**Kaggle notebook:** PASTE_KAGGLE_LINK_HERE
**Result:** ROC-AUC **0.8814** on the competition leaderboard (5-fold cross-validation: 0.880)

## Problem
Binary classification on 68,121 past bookings (17,031 to predict) from a city hotel and a resort hotel in Portugal. The metric is ROC-AUC, so the model is judged on how well it **ranks** bookings by cancellation risk.

## Approach
1. **Cleaning:** fixed inconsistent text and month formats, turned impossible values (negative stays, a 5,400/night price, 10 babies) into missing values, and filled meaningful gaps (no agent, no company, no children).
2. **Features (18 new):** total nights and guests, price per person, booking date (arrival minus lead time), arrival weekday, previous-cancellation rate, how tight the stay is, and frequency encodings for country, agent and company.
3. **Validation:** 5-fold stratified cross-validation, with identical bookings kept in the same fold so the score can't be inflated by near-duplicates.
4. **Models:** LightGBM and CatBoost, combined with a rank-average blend.

## Results
| Model | ROC-AUC |
|---|---|
| LightGBM alone (5-fold CV) | 0.8795 |
| **LightGBM + CatBoost blend (5-fold CV)** | **0.8803** |
| Kaggle leaderboard (submitted blend) | **0.8814** |

The leaderboard score matches cross-validation, which shows the validation was reliable rather than lucky.

## Tools
Python, pandas, NumPy, scikit-learn, LightGBM, CatBoost

## Files
- `hotel_cancellation_solution.ipynb`: full notebook (cleaning → features → model → submission)
