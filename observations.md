# StarSight: Written Observations

## Observations from EDA

1. **Class distribution.** The dataset is balanced, with about 2,000 repositories
   in each class (High 2001, Low 2000, Medium 1999). This comes from the
   sampling design (equal queries per star range), not from GitHub's natural
   distribution, so random guessing would be right about 33% of the time.

2. **Forks and issues rise steeply with popularity.** Median forks_count is 18
   for Low, 131 for Medium and 2,840 for High, and median open issues goes
   from 3 to 17 to 186. These are the clearest separators between the classes.
   forks_count is extremely right-skewed (skew = 7.59), so a log scale was used
   for plotting, whereas age_days is almost symmetric (skew = 0.03).

3. **Popular repositories are actively maintained.** The median High repository
   was pushed 3 days ago, compared with 263 days for Medium and 734 days for
   Low. High repositories are also older (median 3,327 days vs 2,366 for Low),
   which suggests popularity builds up over time but only for projects that
   keep being updated.

4. **Documentation and metadata signals.** The share of repositories with a
   license rises from 74% (Low) to 88% (Medium) to 96% (High), and the share
   with GitHub Pages rises from 12% to 20% to 28%. Median size_kb also grows
   from about 1.5 MB to 5.4 MB to 57 MB, so larger projects tend to be more
   popular. Surprisingly, has_wiki goes the other way, falling from 78% (Low) to
   53% (High), possibly because well-known projects use external documentation
   sites instead of the GitHub wiki.

5. **Correlations are weak in pairs.** The strongest correlation is
   forks_count with open_issues_count (r = 0.35), and all other pairs are below
   0.31 in absolute value, so there is no problem with redundant features. Pearson
   correlation understates the forks signal because of its heavy skew, which is
   why the class-wise medians are more informative.

6. **Limitation: language chart is uninformative.** Each language contributes
   an equal share to each class (about 33%) because of the equal-quota
   collection, so this dataset cannot show whether a language makes a repository
   more popular. Language is still kept as a feature for the models.

Day3 - Observations
Confusion matrix observations

1. Most errors are between neighbouring classes. Of the 89 mistakes, 61 (about 69%) are Low↔Medium (32 Low predicted as Medium, 29 Medium predicted as Low). Another 24 (about 27%) are Medium↔High. Only 4 (about 4%) are Low↔High, and no Low repo was ever predicted as High.

2. The errors follow the star ordering. The model rarely makes large jumps because the class boundaries sit on a continuous star scale. A repo with 95 stars looks almost the same as one with 105, so mistakes cluster near the 100 and 1000 thresholds. These are boundary cases, not wild misclassifications.

3. Per-class performance.

Class	Recall	Precision
Low	92.0% (368/400)	91.8%
Medium	90.8% (363/400)	88.3%
High	95.0% (380/400)	97.9%

High is the easiest class to identify, and Medium is the hardest. Medium sits between two boundaries, so it can be confused in both directions. Medium also has the lowest precision, because it receives mistaken predictions from both sides (32 from Low, 16 from High).

Feature importance observations

4. Forks dominate. forks_count has an importance of about 0.52, more than half of the total. open_issues_count follows at about 0.15, and the two together account for roughly two-thirds (about 0.67) of the model's decisions.

5. Why this is expected, not a bug. Forks and issues are signs of real user engagement, so they rise together with stars. A repo that many people fork and open issues on is usually one many people have starred. This also makes the task easier, which is one of the limitations in your plan.

6. Secondary and weak features. days_since_push (about 0.08), size_kb (about 0.07) and age_days (about 0.05) carry modest signal, so recently maintained, larger and older repos lean toward higher classes. language and description_length add a little (about 0.04 each). topics_count, has_license, has_wiki and has_pages contribute almost nothing (about 0.02 or less), so these simple metadata flags barely help predict popularity.

1. Best model. XGBoost performed best on the held-out test set, with accuracy 92.58% and macro F1 0.9262. It was followed by Random Forest (macro F1 0.9220) and Logistic Regression (0.9216). XGBoost was therefore selected and saved as best_model.joblib.

2. The models are very close. The lead is only about 0.4 percentage points in macro F1 (0.9262 vs 0.9220 vs 0.9216). On a test set of this size, that is only a few repositories, so the three models should be described as comparable, not as XGBoost being clearly superior. XGBoost was chosen because it ranked first on the agreed criterion, macro F1.

3. Logistic Regression nearly matches the tree models. The baseline scored 0.9216 against 0.9262 for XGBoost. This suggests the popularity classes are largely separable by a roughly linear boundary once skewed features (forks, issues, size) are log-transformed. The extra flexibility of Random Forest and XGBoost bought very little here.

4. Balanced performance across classes. Accuracy, macro precision and macro recall are almost identical for every model (for example XGBoost: 0.9258, 0.9268, 0.9258). That indicates no class is being sacrificed to inflate the score, which fits with the star-range sampling that kept classes balanced.

5. Sanity check on the high score. Around 92% is high but not suspicious (the warning threshold in the plan is about 97%), as long as stargazers_count and watchers_count are not in FEATURES. Forks and open issues track stars closely, so a strong score is expected. Mention this in your limitations section.

## 1. Which model won, and why
XGBoost won on the held-out test set with accuracy 92.58% and macro F1 0.9262,
narrowly ahead of Random Forest (0.9220) and Logistic Regression (0.9216).
The gap is only about 0.4 percentage points, so the three models are effectively
comparable. XGBoost was selected because it ranked first on macro F1, the agreed
selection criterion. Logistic Regression nearly matching the tree models suggests
the classes are largely separable once skewed features are log-transformed.

## 2. Which classes were confused most
Of 89 errors on 1,200 test repos, 61 (about 69%) were Low <-> Medium and 24 (about 27%)
were Medium <-> High. Only 4 were Low <-> High, and no Low repo was predicted as High.
Errors follow the star ordering and cluster near the 100 and 1000 star boundaries.
High was the easiest class (recall 95.0%) and Medium the hardest (recall 90.8%,
precision 88.3%), as it sits between two boundaries.

## 3. Top features and what they mean
- forks_count (about 0.52 importance): forks signal real community engagement and rise together with stars.
- open_issues_count (about 0.15): active issue discussion indicates many users.
- days_since_push (about 0.08): recently maintained repos lean toward higher classes.
Flags such as has_license, has_wiki and has_pages contributed almost nothing.
Importance shows what the model relies on, not what causes popularity.

## 4. Forks ablation (retrained without forks_count)
- Macro F1 with forks_count: 0.9262
- Macro F1 without forks_count: ____
- Drop: ____
Interpretation: ____

## 5. Honest limitations
- The dataset was sampled by star range, so class balance is artificial and does
  not reflect the real distribution of GitHub repositories.
- Popularity depends on factors the features cannot see (marketing, a viral post,
  author fame).
- Forks and open issues correlate strongly with stars, which is natural but makes
  the task easier than it would otherwise be.