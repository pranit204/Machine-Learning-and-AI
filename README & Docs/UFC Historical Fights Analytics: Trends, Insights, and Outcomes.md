**Project Overview**
This project analyzes historical UFC fight data to uncover patterns, trends, and insights into fight outcomes. It evaluates the performance of betting favorites between 2010 and 2020, performs statistical analyses, and builds a machine learning model to predict whether the betting favorite will win based on odds data.

**Code Functionality**
Data Cleaning and Filtering:
Ensures critical columns such as R_odds, B_odds, and Winner are present in the dataset.
Converts the date column into a proper datetime format.
Filters the dataset for fights that occurred between 2010 and 2020.

Betting Analysis:
Calculates the percentage of times the betting favorite won during the specified period by:
Identifying the favorite using the minimum odds (R_odds or B_odds).
Comparing the favorite’s odds with the fight winner to determine success.
Outputs the favorite win rate, providing a key metric for evaluating betting market accuracy.

Visualizations:
Generates insights with the following plots:
Favorite vs. Underdog Success Rate: A bar chart showing how often favorites win compared to underdogs.
Weight Class Analysis: A bar chart displaying the distribution of wins across different weight classes.
Win Method Analysis (if available): A pie chart showing the percentage of wins by KO/TKO, decision, or submission.

Top Fighter Analysis:
Identifies the top 10 fighters by match count based on appearances in the dataset.

Predictive Modeling:
Implements a Logistic Regression model to predict whether the betting favorite will win based on:
Favorite Odds: The lowest odds between the two fighters.
Underdog Odds: The higher odds between the two fighters.
Evaluates the model with accuracy metrics and outputs its performance.

Results Export:
Saves the cleaned and enriched dataset, including analysis and predictions, to a CSV file (ufc_analysis_results.csv).

**Key Results**
Betting Favorite Win Rate:
The analysis shows the percentage of fights won by the betting favorite between 2010 and 2020.
Example Output:
**Percentage of times the betting favorite won (2010-2020): 67.35%**

Model Accuracy:
The Logistic Regression model achieves an accuracy of ~66% for predicting the success of betting favorites.

Top Fighters by Match Count:
Lists the fighters with the most appearances in the dataset during the analysis period.

Visualizations
Favorite vs. Underdog Success Rate:
A bar chart highlighting the win rates for favorites vs. underdogs.
