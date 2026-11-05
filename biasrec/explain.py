"""Explanation layer: turns score components into human-readable reasons."""

from __future__ import annotations

import numpy as np

from .bias import desc_order
from .nlp import EMOTIONS


def _pct(x: float) -> str:
    return f"{round(100 * x)}%"


def _join(items: list[str]) -> str:
    items = [i for i in items if i]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


class Explainer:
    def __init__(self, engine):
        self.e = engine

    def bias_evidence(self, key: str, profile) -> str:
        d, e = profile.details, self.e
        if key == "negativity":
            return (f"{_pct(d['negative_share'])} of your history has a dark or negative tone "
                    f"(typical viewer: {_pct(e.bias.typical('negativity'))}).")
        if key == "novelty_avoidance":
            top = self.top_genres(profile, 2)
            return (f"Your viewing is concentrated in {_join(top)}, genre diversity is "
                    f"{_pct(d['genre_entropy'])} of the maximum, and {_pct(d['mainstream'])} "
                    "of your picks are mainstream hits.")
        if key == "confirmation":
            themes = [e.features["theme_names"][t] for t in d["top_themes"][:2]]
            gap = d["theme_rating_gap"]
            extra = (f" You rate those themes {gap:.1f}★ higher than everything else."
                     if gap > 0.15 else "")
            return f"You keep returning to themes like “{themes[0]}”." + extra
        if key == "recency":
            drift = d["rating_drift"]
            trend = (f", and your ratings drift upward over time (ρ={drift:.2f})" if drift >= 0.1
                     else ", and your ratings stay steady over time")
            if profile.timestamps is None:
                return "Watch dates are needed to measure recency bias; profiles built here don't have them."
            return (f"{_pct(d['fresh_release_share'])} of your watches were new releases at the "
                    f"time{trend}.")
        if key == "hero_projection":
            strong = d["scores"]["hero_projection"] >= 0.5
            return (f"{_pct(d['hero_share'])} of what you watch centres on a hero or saviour arc"
                    + (", and you rate those films highly." if strong else "."))
        if key == "comfort_zone":
            return (f"Your comfort-zone index is {d['comfort_index']:.2f} "
                    f"(typical viewer: {e.bias.typical('comfort_zone'):.2f}); lower means your "
                    "films sit close together in story space.")
        return ""

    def top_genres(self, profile, n=3) -> list[str]:
        counts = profile.details["genre_counts"]
        order = desc_order(counts)[:n]
        return [self.e.features["genres"][i] for i in order if counts[i] > 0]

    def recommendation(self, m: int, profile, comp: dict, bias_contrib: dict,
                       used: dict | None = None) -> dict:
        e = self.e
        used = used or {}
        mv = e.movies.iloc[m]
        title = mv["clean_title"]
        anchor = e.cf.most_similar_rated(m, profile.idx, profile.ratings)
        anchor_title = e.movies.iloc[anchor[0]]["clean_title"] if anchor else None

        main_bias = max(bias_contrib, key=bias_contrib.get) if bias_contrib else None
        if main_bias and bias_contrib[main_bias] < 0.06:
            main_bias = None
        variant = used.get(main_bias, 0) if main_bias else 0

        valence = float(e.features["valence"][m])
        lead_emotion = EMOTIONS[int(np.argmax(e.features["emotions"][m]))]
        genres = list(mv["genre_list"])
        user_top = self.top_genres(profile, 3)
        new_genres = [g.lower() for g in genres if g not in user_top]
        decade = f"{int(mv['year']) // 10 * 10}s" if not np.isnan(mv["year"]) else "an earlier era"
        theme = e.features["theme_names"][int(e.features["themes"][m].argmax())]
        tone = "hopeful" if valence > 0.25 else "lighter" if valence > 0 else "less bleak"
        like = f" while keeping the depth of {anchor_title}" if anchor_title else ""

        options = {
            "negativity": [
                f"You often watch dark, heavy films. {title} offers a {tone} story led by {lead_emotion}{like}.",
                f"Most of your history is intense. This one runs on {lead_emotion} instead, a real change of emotional register.",
                f"A {tone} counterweight to your darker picks, without losing substance.",
            ],
            "novelty_avoidance": [
                f"You rarely step outside {_join(user_top[:2])}. This {_join(new_genres[:2])} pick from the {decade} widens your range without feeling disconnected."
                if new_genres else f"A less-travelled title from the {decade} that people with your taste rate well.",
                f"{title} brings {_join(new_genres[:2])} into a history that is mostly {_join(user_top[:2])}."
                if new_genres else "Fewer people have seen this one, but it rewards viewers with your taste.",
                f"A {decade} film in a direction your history hasn't taken yet.",
            ],
            "confirmation": [
                f"It explores “{theme}”, a perspective that rarely appears in your history, so it challenges rather than confirms your usual worldview.",
                f"Built around “{theme}”, which your ratings barely touch. Worth seeing the world from that angle.",
            ],
            "recency": [
                f"You lean toward new releases. This {decade} film has stood the test of time and shows you a different era of filmmaking.",
                f"Older than almost everything you rate, and a reminder that the {decade} still have something to say.",
            ],
            "hero_projection": [
                "Instead of a lone hero saving the day, this story is about ordinary, flawed people: a mirror rather than an ideal.",
                "No chosen one here. The drama comes from people making hard choices at human scale.",
            ],
            "comfort_zone": [
                "It sits just outside your usual cluster of films: familiar enough to enjoy, different enough to stretch you.",
                f"Different territory in story space, connected to your taste through {anchor_title}." if anchor_title
                else "Different territory in story space, but close enough to your taste to land.",
                "A step sideways from what you usually watch: new ground, similar craft.",
            ],
        }
        if main_bias:
            choices = options[main_bias]
            reason = choices[variant % len(choices)]
        else:
            reason = "A strong match for your taste" + (f", close to {anchor_title}." if anchor_title else ".")

        quality = float(mv["quality"])
        if anchor_title and main_bias:
            fit = f"Why it still fits: viewers who liked {anchor_title} rate it highly ({quality:.1f}★ average)."
        else:
            fit = f"Rated {quality:.1f}★ on average by {int(mv['n_ratings'])} viewers."
        stretch = float(comp["stretch"])
        role = "Bridge" if stretch < 0.42 else ("Stretch" if stretch > 0.62 else "Expand")
        return {"reason": reason, "fit": fit, "counters": main_bias, "role": role}
