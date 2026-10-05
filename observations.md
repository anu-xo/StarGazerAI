# StarSight: Written Observations

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