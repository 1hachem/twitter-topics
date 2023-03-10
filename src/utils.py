import pickle

from sklearn.model_selection import train_test_split


def save_pickle(obj, path):
    """Save an object to a pickle file"""
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    """Load an object from a pickle file"""
    with open(path, "rb") as f:
        return pickle.load(f)


def split_data(x, y, test_size=0.2, val_size=0.1, random_state=42):
    """Split data into train, test and val sets"""
    X_train, x_test, Y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )
    x_train, x_val, y_train, y_val = train_test_split(
        X_train, Y_train, test_size=val_size, random_state=random_state
    )
    return x_train, x_val, x_test, y_train, y_val, y_test
