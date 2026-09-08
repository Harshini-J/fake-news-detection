# User Study Results

## Design Summary

- **Participants:** 12, non-technical
- **Materials:** 10 LIAR test-set statements (5 correctly classified, 3 incorrectly
  classified, 2 baseline/DistilBERT disagreement cases), each shown via the project's
  Streamlit dashboard with the model's prediction, confidence score, and
  Gemini-generated explanation
- **Measures:** 5-point Likert ratings for clarity and trust per example, plus a
  closing preference question (plain-English explanation vs. raw SHAP word list)

## Likert Ratings (n=12)

| Example | Clarity M (SD) | Trust M (SD) |
|---------|----------------|--------------|
| 1       | 4.08 (0.79)    | 3.92 (0.90)  |
| 2       | 4.17 (0.58)    | 3.83 (0.94)  |
| 3       | 3.67 (0.89)    | 3.42 (0.79)  |
| 4       | 4.25 (0.62)    | 4.00 (0.60)  |
| 5       | 3.92 (0.79)    | 3.67 (0.78)  |
| 6       | 3.58 (0.79)    | 3.25 (0.97)  |
| 7       | 3.50 (0.67)    | 3.17 (0.72)  |
| 8       | 3.42 (0.90)    | 3.08 (1.00)  |
| 9       | 4.08 (0.51)    | 3.83 (1.11)  |
| 10      | 4.33 (0.49)    | 4.08 (0.67)  |
| **Overall** | **3.90 (0.76)** | **3.62 (0.90)** |

## Explanation Format Preference (n=12)

| Preference                 | Participants | Percentage |
|-----------------------------|:------------:|:----------:|
| Plain-English explanation   | 10           | 83.3%      |
| Raw SHAP word list          | 1            | 8.3%       |
| No strong preference        | 1            | 8.3%       |

## Key Findings

1. Both clarity (3.90/5) and trust (3.62/5) ratings were comfortably above the
   5-point scale midpoint, supporting the value of translating attribution scores
   into natural-language explanations.

2. 83.3% of participants explicitly preferred the plain-English explanation over
   the raw SHAP word-importance list.

3. The two lowest-rated examples (7 and 8) corresponded to the model's two
   lowest-confidence predictions (60% and 50%), suggesting explanation clarity is
   perceived as lower when the underlying prediction is itself more uncertain —
   even when the explanation faithfully communicates that uncertainty.

## Limitations

- Convenience sample (n=12), not demographically representative or task-diverse
- Measures preference/clarity ratings rather than task-based comprehension
  (e.g., "can the participant correctly explain why the model made this call?")
- A full conference-level study would need n=50+ across distinct user groups
  (journalists, general readers, ML practitioners)
