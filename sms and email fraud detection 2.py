import re
import numpy as np
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

class SecurityFeatures(BaseEstimator, TransformerMixin):
    URGENT = re.compile(r'urgent|verify|winner|claim|free|act now|congratulations', re.I)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for m in X:
            n = max(len(m), 1)
            rows.append([
                len(m),
                sum(c.isupper() for c in m) / n,
                sum(c.isdigit() for c in m) / n,
                m.count('!'),
                len(re.findall(r'https?://|www\.', m, re.I)),
                len(re.findall(r'[$£€₹]', m)),
                len(re.findall(r'\b\d{5,}\b', m)),
                int(bool(self.URGENT.search(m))),
            ])
        return np.array(rows, dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.array([
            'length',
            'uppercase_ratio',
            'digit_ratio',
            'exclamation_count',
            'link_count',
            'currency_count',
            'long_number_count',
            'urgency_keyword'
        ], dtype=object)

df = pd.read_csv('sms.tsv', sep='\t', header=None, names=['label', 'message'])
df['label'] = (df['label'] == 'spam').astype(int)

msg_train, msg_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2,
    random_state=42, stratify=df['label']
)

features = FeatureUnion([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, sublinear_tf=True
    )),
    ('security', Pipeline([
        ('extract', SecurityFeatures()),
        ('scale', StandardScaler())
    ]))
])

pipe = Pipeline([
    ('features', features),
    ('clf', LogisticRegression(
        class_weight='balanced', max_iter=1000
    ))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_validate(
    pipe, msg_train, y_train, cv=cv,
    scoring=['accuracy', 'precision', 'recall', 'f1']
)

for m in ['accuracy', 'precision', 'recall', 'f1']:
    s = scores[f'test_{m}']
    print(f"CV {m:<9}: {s.mean():.4f} (+/- {s.std():.4f})")

grid = GridSearchCV(
    pipe,
    param_grid={
        'clf__C': [0.1, 1, 10],
        'features__tfidf__ngram_range': [(1, 1), (1, 2)]
    },
    cv=cv, scoring='f1', n_jobs=-1
)

grid.fit(msg_train, y_train)
best = grid.best_estimator_

print("\nBest params:", grid.best_params_)

THRESHOLD = 0.5

probs = best.predict_proba(msg_test)[:, 1]
y_pred = (probs >= THRESHOLD).astype(int)

print("\n", classification_report(
    y_test, y_pred, target_names=['Normal', 'Fraud']
))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

errors = msg_test[y_pred != y_test]
print(f"\n{len(errors)} mistakes. First 5:")

for text, true in zip(errors.head(5), y_test[errors.index].head(5)):
    print(f"  true={true} | {text[:70]}")

names = best.named_steps['features'].get_feature_names_out()
coefs = best.named_steps['clf'].coef_[0]

top = np.argsort(coefs)[-10:][::-1]

print("\nTop fraud indicators:")

for i in top:
    print(f"  {names[i]:<30} {coefs[i]:.3f}")

joblib.dump(best, 'fraud_model.pkl')
print("\nModel saved as fraud_model.pkl")

print("\nPaste a message. Type 'quit' to exit.\n")

while True:
    msg = input("Enter message: ").strip()

    if msg.lower() in ('quit', 'exit', 'q'):
        break

    if not msg:
        continue

    p = best.predict_proba([msg])[0][1]

    print(
        f"Result: {'FRAUD' if p >= THRESHOLD else 'NORMAL'}"
        f" | Fraud probability: {p * 100:.2f}%\n"
    )
