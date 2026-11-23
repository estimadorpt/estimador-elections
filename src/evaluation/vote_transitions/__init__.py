"""How the vote moved between legislative elections, from the survey side.

Survey transition matrices (EEP recalled vote, Católica 2025 exit poll) raked to the official mainland
margins, parish bounds, the survey odds ratios fitted to every parish result, a parish model with the
survey as extra likelihood (dm_hybrid, needs torch), vote by group across EEP waves, and group vote rates
by district for the seat simulator (group_rates). Run with `python -m src.evaluation.vote_transitions OUT_DIR`.
"""
