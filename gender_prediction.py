import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv("train.csv", sep=";")
labels = pd.read_csv("train_labels.csv", sep=";")
ref_vec = pd.read_csv("referer_vectors.csv", sep=";")
geo = pd.read_csv("geo_info.csv", sep=";")
test = pd.read_csv("test.csv", sep=";")
test_ids = pd.read_csv("test_users.csv", sep=";")

train = train.merge(labels, on="user_id")
train = train.merge(ref_vec, on="referer", how="left")
train = train.merge(geo, on="geo_id", how="left")
test = test.merge(ref_vec, on="referer", how="left")
test = test.merge(geo, on="geo_id", how="left")

ua_all = pd.concat([train["user_agent"], test["user_agent"]])
tz_all = pd.concat([train["timezone_id"], test["timezone_id"]])
ua_enc = LabelEncoder().fit(ua_all)
tz_enc = LabelEncoder().fit(tz_all)
train["user_agent"] = ua_enc.transform(train["user_agent"])
test["user_agent"] = ua_enc.transform(test["user_agent"])
train["timezone_id"] = tz_enc.transform(train["timezone_id"])
test["timezone_id"] = tz_enc.transform(test["timezone_id"])

train.drop(columns=["user_id", "request_ts", "referer"], inplace=True)
test.drop(columns=["request_ts", "referer"], inplace=True)

X = train.drop(columns=["target"]).select_dtypes(include=["number"])
y = train["target"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

test_grouped = test.groupby("user_id").mean(numeric_only=True).reset_index()
test_final = test_ids.merge(test_grouped, on="user_id", how="left")
X_test = test_final.drop(columns=["user_id"])
pred = model.predict(X_test)

pd.DataFrame({"user_id": test_final["user_id"], "target": pred}).to_csv("submission.csv", index=False)
print("Готово")
