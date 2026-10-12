"""Build viz/sim_data.js for generation_sim.html from data/processed/sm_generations.csv.

Each era (2018, 2024) gets a weighted sample of adult respondents. For every question
step we store which answer bucket each sampled person falls in, plus exact weighted
percentages from the full survey (overall and by generation) for the labels.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
N_DOTS = 800
SEED = 2018

GENERATIONS = ["Gen Z", "Millennials", "Gen X", "Boomers & older"]


def agree(col, na_label="Not on social media"):
    def f(df):
        s = df[col]
        out = pd.Series(na_label, index=df.index, dtype=object)
        out[s >= 3] = "Agree"
        out[s.isin([1, 2])] = "Disagree"
        return out
    return f


def selected(col, yes="Yes", no="No"):
    def f(df):
        out = pd.Series(no, index=df.index, dtype=object)
        out[df[col] == 1] = yes
        out[df["sm_not_a_user"] == 1] = "Not on social media"
        return out
    return f


def mental_health(df):
    s = df["mental_health_rating"]
    return pd.Series(np.select([s >= 4, s == 3, s <= 2], ["Very good / Excellent", "Good", "Fair / Poor"], None), index=df.index)


def stress(df):
    s = df["stress_level"]
    return pd.Series(np.select([s >= 8, s >= 4, s >= 1], ["High (8-10)", "Moderate (4-7)", "Low (1-3)"], None), index=df.index)


def check_freq(df):
    s = df["sm_check_freq"]
    return pd.Series(np.select(
        [s == 5, s == 4, s == 3, s.isin([1, 2])],
        ["Constantly", "Often", "Sometimes", "Rarely / Never"], "Not on social media"), index=df.index)


def internet_freq(df):
    s = df["internet_use_freq"]
    return pd.Series(np.select(
        [s == 1, s == 2, s >= 3],
        ["Almost constantly", "Several times a day", "Once a day or less"], None), index=df.index)


# (theme, 2018 question, 2018 bucketer, 2018 bucket order, 2024 question, 2024 bucketer, 2024 order, same_question)
AGREE = ["Agree", "Disagree", "Not on social media"]
YESNO = ["Yes", "No", "Not on social media"]
STEPS = [
    ("How plugged in?",
     "How often do you check social media?", check_freq,
     ["Constantly", "Often", "Sometimes", "Rarely / Never", "Not on social media"],
     "How often do you use the internet?", internet_freq,
     ["Almost constantly", "Several times a day", "Once a day or less"], False),
    ("Self-image",
     "Social media makes me feel bad about myself.", agree("sm_feel_bad_about_self"), AGREE,
     "I find that I compare myself to others when using social media.", selected("sm_compare_self"), YESNO, False),
    ("Losing time",
     "I worry about the negative effects of social media on my physical and mental health.", agree("sm_worry_health_effects"), AGREE,
     "I find myself filling my time scrolling through social media without realizing it.", agree("sm_mindless_scrolling", None), AGREE[:2], False),
    ("Politics online",
     "Political and cultural discussions on social media cause me stress.", agree("sm_political_discussions_stress"), AGREE,
     "Social media content about politics has negatively impacted my mental health.", selected("sm_politics_hurt_mental_health"), YESNO, False),
    ("Support",
     "Social media makes me feel supported.", agree("sm_feel_supported"), AGREE,
     "I find that social media can be a helpful way to de-stress.", agree("sm_helpful_destress", None), AGREE[:2], False),
    ("Mental health",
     "In general, would you say your mental health is…", mental_health, ["Very good / Excellent", "Good", "Fair / Poor"],
     "In general, would you say your mental health is…", mental_health, ["Very good / Excellent", "Good", "Fair / Poor"], True),
    ("Stress",
     "Average stress level in the past month, 1-10", stress, ["Low (1-3)", "Moderate (4-7)", "High (8-10)"],
     "Average stress level in the past month, 1-10", stress, ["Low (1-3)", "Moderate (4-7)", "High (8-10)"], True),
]


def wpct(buckets, weights, order):
    total = weights[buckets.notna()].sum()
    return [round(float(weights[buckets == b].sum() / total * 100), 1) for b in order]


def build_era(df, year, rng):
    era = df[(df["year"] == year) & ~df["is_teen"]].copy()
    era["gen"] = era["generation"].replace({"Silent": "Boomers"}).map(
        {"Gen Z": 0, "Millennials": 1, "Gen X": 2, "Boomers": 3})
    idx = 1 if year == 2018 else 4
    steps_b = []
    for step in STEPS:
        fn, order = step[idx + 1], step[idx + 2]
        b = fn(era)
        bad = set(b.dropna()) - set(order)
        assert not bad, (step[0], year, bad)
        steps_b.append((b, order))
    # Only sample people with an answer on every step
    complete = np.all([b.notna().to_numpy() for b, _ in steps_b], axis=0)
    pool = era[complete]
    p = pool["weight"] / pool["weight"].sum()
    pick = rng.choice(pool.index.to_numpy(), size=N_DOTS, replace=True, p=p.to_numpy())

    steps_out = []
    for step, (b, order) in zip(STEPS, steps_b):
        steps_out.append({
            "question": step[idx],
            "buckets": order,
            "assign": "".join(str(order.index(x)) for x in b.loc[pick]),
            "pct": wpct(b, era["weight"], order),
            "pctByGen": [wpct(b[era["gen"] == g], era.loc[era["gen"] == g, "weight"], order) for g in range(4)],
        })
    return {
        "year": year,
        "nRespondents": int(len(era)),
        "nByGen": [int((era["gen"] == g).sum()) for g in range(4)],
        "gens": "".join(str(int(g)) for g in era.loc[pick, "gen"]),
        "steps": steps_out,
    }


def main():
    df = pd.read_csv(ROOT / "data/processed/sm_generations.csv", low_memory=False)
    rng = np.random.default_rng(SEED)
    data = {
        "generations": GENERATIONS,
        "themes": [s[0] for s in STEPS],
        "sameQuestion": [s[7] for s in STEPS],
        "eras": [build_era(df, 2018, rng), build_era(df, 2024, rng)],
    }
    out = ROOT / "viz/sim_data.js"
    out.write_text("window.SIM_DATA = " + json.dumps(data, separators=(",", ":")) + ";\n")
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")
    for era in data["eras"]:
        print(era["year"], era["nRespondents"], era["nByGen"])
        for t, s in zip(data["themes"], era["steps"]):
            print("  ", t, dict(zip(s["buckets"], s["pct"])))


if __name__ == "__main__":
    main()
