from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib
X = [
 "anxiety stress fatigue worry panic",
 "sadness loneliness isolation crying",
 "happy positive energetic motivation",
 "anger frustration irritability aggression",
 "confusion indecision forgetfulness",
 "calm relaxed peaceful mindfulness meditation"
 ]
y = [0,1,2,3,4,5]
pipeline = Pipeline([
 ("vectorizer", TfidfVectorizer()),
 ("model", RandomForestClassifier(n_estimators=50, random_state=42))
 ])
pipeline.fit(X, y)
joblib.dump(pipeline, "model/ml_model_6groups.pkl")
exit()