from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pickle
import csv
from pathlib import Path


class TierClassifier:
    """ML-based query-to-tier classifier using TF-IDF + logistic regression."""

    def __init__(self):
        self.model = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=100, ngram_range=(1, 2))),
            ("classifier", LogisticRegression(max_iter=200, random_state=42))
        ])
        self.classes = ["fast", "general", "reasoning"]
        self.is_trained = False

    def train(self, queries: list[str], labels: list[str]) -> None:
        """Train classifier on labeled queries."""
        self.model.fit(queries, labels)
        self.is_trained = True

    def predict(self, query: str) -> tuple[str, float]:
        """
        Predict tier and confidence for a query.
        Returns: (predicted_tier, confidence_score)
        """
        if not self.is_trained:
            raise RuntimeError("Classifier not trained. Call train() first.")
        
        tier = self.model.predict([query])[0]
        confidence = self.model.predict_proba([query]).max()
        return tier, confidence

    def save(self, filepath: str) -> None:
        """Persist model to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self, filepath: str) -> None:
        """Load model from disk."""
        if Path(filepath).exists():
            with open(filepath, 'rb') as f:
                self.model = pickle.load(f)
            self.is_trained = True


def load_training_data(csv_path: str = None) -> tuple[list[str], list[str]]:
    """Load training data from CSV file. Format: query,tier"""
    if csv_path is None:
        # Default path relative to this file
        current_dir = Path(__file__).parent.parent.parent
        csv_path = current_dir / "datasets" / "training_queries.csv"
    
    queries = []
    labels = []
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Remove quotes if present
                query = row['query'].strip().strip('"')
                tier = row['tier'].strip().strip('"')
                queries.append(query)
                labels.append(tier)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Training dataset not found at {csv_path}. "
            f"Please ensure backend/datasets/training_queries.csv exists."
        )
    
    return queries, labels


def create_default_classifier() -> TierClassifier:
    """Create and train a classifier with data from CSV file."""
    queries, labels = load_training_data()
    classifier = TierClassifier()
    classifier.train(queries, labels)
    return classifier