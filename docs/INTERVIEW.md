# Interview kit — "Who Will Respond?"

## Resume bullets (pick one)

**Full version**
> Built an end-to-end uplift-modeling system that predicts who responds *because of* a marketing offer
> (S/T/X/R meta-learners, class transformation, uplift random forest, causal forest; LightGBM, causalml,
> econml). On a 64k-customer randomized experiment, the best models doubled campaign profit vs. mailing
> everyone (+$31–39k per 100k customers, ~50% fewer offers) while response-model targeting was no better
> than random (Qini 0.011 vs 0.07); validated with 5-fold CV, bootstrap CIs, a 500k-row Criteo replication,
> and shipped a Streamlit ROI simulator with SHAP explanations.

**Short version**
> Uplift modeling project (causal ML): 7 CATE estimators vs. response-model baselines; Qini 0.07 vs 0.01;
> ~2× campaign profit at 50% of the mailing cost; Streamlit app with ROI simulator + SHAP; MLflow, pytest.

## 2-minute pitch

> **The problem.** Companies send offers to everyone, but most people either buy anyway or never will.
> Paying to contact them is waste — and some customers react badly. The real question isn't "who will
> buy?" but "who will buy *because* we contacted them?" That difference is the *uplift*.
>
> **The data.** I used the Hillstrom e-mail experiment — 64,000 customers randomly assigned to get an
> e-mail or not. Randomization is what makes it causal, so I verified it first: covariates were balanced
> and a model trying to predict who got the e-mail scored AUC 0.50 — pure chance. The average effect was
> +4.5 points on visits, but it ranged from +1 to +7 points across customer groups. That spread is the
> money uplift modeling goes after.
>
> **The models.** Uplift is never observed for one person — you either sent the e-mail or you didn't —
> so I estimated it with seven methods: S, T, X and R meta-learners on LightGBM, class transformation,
> an uplift random forest, and a causal forest. I benchmarked them against a normal response model,
> using Qini and uplift curves, 5-fold cross-validation, and bootstrap confidence intervals on a held-out
> test set. I tuned the shared LightGBM settings by cross-validation only, never on test.
>
> **The result.** Every uplift model beat every baseline. The response model — the thing most teams would
> build — was statistically no better than random, because it targets people who buy anyway. In money
> terms, at $1 per offer and $30 per response, targeting the top half by uplift earned about twice the
> profit of mailing everyone while sending half the offers. A what-if analysis showed that above about $3
> per offer the right answer is not to run the campaign at all — which the model tells you.
>
> **Honesty checks.** I tested on simulated data where the true uplift is known, across five random seeds,
> on the men's e-mail arm, and on a 500k-row Criteo sample. One finding I like to share: on the men's
> e-mail, uplift models found almost no heterogeneity — the e-mail works about equally for everyone, so
> mass mailing is fine there. Uplift modeling pays off only when effects really differ between customers.
>
> **The product.** I shipped a Streamlit dashboard: model comparison, an ROI simulator with cost and
> value sliders, a segment explorer, and a "should we send this customer the offer?" tool with SHAP
> reasons. Everything is tracked in MLflow and covered by 47 tests.

## Questions you should be ready for

**Why not just a classifier?** It ranks by P(buy), so the top of the list is Sure Things who buy anyway.
In my test set the response model had Qini 0.011 — the same as random (0.010).

**What's the fundamental problem of causal inference?** You see only one potential outcome per person, so
individual uplift can't be measured; we estimate the CATE for people with similar features.

**Why does randomization matter?** It makes treatment independent of the potential outcomes, so
treated − control differences are causal. I checked it with SMDs (max 0.009) and a propensity AUC of 0.497.

**S vs T vs X vs R learner?**
- S: one model with treatment as a feature; can under-use treatment → shrinks uplift toward 0.
- T: separate models per group; the difference of two noisy models is noisy.
- X: imputes individual effects with the other group's model, then models them directly; weighting by
  propensity helps with unbalanced groups (Criteo is 85% treated).
- R: Robinson residual-on-residual regression with cross-fitting; targets the CATE directly and is robust
  to errors in the outcome model.

**What is the Qini coefficient?** Area between the model's Qini curve and the random line, normalised by a
perfect model's area. 0 = random, 1 = perfect. Hillstrom values (~0.07) look small, but they're typical
for this weak-signal dataset and the CIs exclude zero.

**How did you choose the best model?** By test Qini, but the top four have overlapping CIs, so I treat them
as tied and deploy the fast, lightweight S-learner in the app. Causal forest had the best and most stable CV score.

**Why surrogate SHAP?** Meta-learners combine several models, so I fit one LightGBM to mimic predicted
uplift (R² 0.91–0.98) and explain that. Top driver: past purchase of women's products (+), men's (−),
rural customers (−).

**So are response models always wrong?** No — on Criteo the response model tied the best uplift model
(Qini 0.111 vs 0.113) because likely visitors were also the ones the ad moved. On Hillstrom they diverge
and the response model fails. Uplift models were never worse in any setting, so they're the safe default.

**Limitations?** Weak signal and few features; "visit" is a proxy; economics are assumed; segment labels are
estimates; uplift-at-k for small k is noisy; no time drift because the data is a single campaign.
