import pandas as pd
from sklearn.model_selection import train_test_split

__all__ = [
    "map_sentiment",
    "load_dataset",
    "train_test_split_sentiment",
]


def map_sentiment(rating: int) -> int:
    """
    Map original 1–5 rating into 3 sentiment classes:
    0 -> negative (1–2)
    1 -> neutral (3)
    2 -> positive (4–5)
    """
    if rating <= 2:
        return 0
    elif rating == 3:
        return 1
    return 2


def load_dataset(path: str) -> pd.DataFrame:
    """Load the Yelp dataset from a CSV file."""
    return pd.read_csv(path)


def train_test_split_sentiment(
    df: pd.DataFrame,
    text_column: str = "Review Text",
    rating_column: str = "Rating",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Convenience helper: map ratings to sentiments and split into train/test.
    Returns: X_train_raw, X_test_raw, y_train, y_test
    """
    y = df[rating_column].apply(map_sentiment)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df[text_column],
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    return X_train_raw, X_test_raw, y_train, y_test

