import re

import numpy as np
import pandas as pd

def pretty_formula(formula):
    return formula.lower()



def parse_composition(text, features, normalize=True):
    if not text or not text.strip():
        raise ValueError("Digite uma composição, por exemplo: 30SiO2+70Na2O")

    feature_lookup = {feature.lower(): feature for feature in features}
    values = {feature: 0.0 for feature in features}
    pattern = re.compile(r"(\d+(?:[.,]\d+)?)\s*([A-Za-z][A-Za-z0-9]*)\s*%?")

    for term in text.split("+"):
        term = term.strip()
        match = pattern.fullmatch(term)
        if not match:
            raise ValueError(f"Termo inválido: {term}")
        amount = float(match.group(1).replace(",", "."))
        formula = match.group(2).lower()
        if formula not in feature_lookup:
            supported = ", ".join(pretty_formula(feature) for feature in features)
            raise ValueError(f"Óxido não reconhecido: {match.group(2)}. Opções: {supported}")
        if amount < 0:
            raise ValueError("As porcentagens não podem ser negativas")
        values[feature_lookup[formula]] += amount

    total = sum(values.values())
    if total <= 0:
        raise ValueError("A soma da composição deve ser maior que zero")

    was_normalized = normalize and not np.isclose(total, 100.0)
    if normalize:
        values = {feature: value * 100.0 / total for feature, value in values.items()}

    frame = pd.DataFrame([values], columns=features)
    return frame, total, was_normalized


def format_composition(row, features, threshold=1e-8):
    components = [
        (float(row[feature]), pretty_formula(feature))
        for feature in features
        if float(row[feature]) > threshold
    ]
    components.sort(reverse=True)
    return "+".join(f"{value:.4g}{formula}" for value, formula in components)
