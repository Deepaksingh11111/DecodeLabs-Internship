import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

class IrisClassificationPipeline:
    def __init__(self, n_neighbors: int = 5, test_size: float = 0.20, random_state: int = 42) -> None:
        """
        Initializes the Supervised Learning Pipeline hyperparameters.
        Sets up the KNN target neighborhood and the historical train-test boundaries.
        """
        self.n_neighbors = n_neighbors
        self.test_size = test_size
        self.random_state = random_state
        
        # Pipeline components
        self.scaler = StandardScaler()
        self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        
        # Data placeholders
        self.X_train, self.X_test = None, None
        self.y_train, self.y_test = None, None

    def execute_input_stage(self) -> None:
        """
        STAGE 1: INPUT & PREPROCESSING
        Loads the Iris benchmark database and normalizes feature dimensions.
        Ensures a mean of 0 and variance of 1 to remove calculation scale biases.
        """
        print("[1/3] Initiating Input Stage: Fetching Iris Benchmark Dataset...")
        iris = load_iris()
        
        # Extract features (Sepal Length, Sepal Width, Petal Length, Petal Width)
        X = iris.data
        y = iris.target
        self.target_names = iris.target_names
        
        print(f" -> Raw Data Found: {X.shape[0]} balanced samples across {X.shape[1]} dimensions.")
        
        # Structural Integrity Split: 80% Training Set, 20% Testing Validation Set
        # Shuffles data before splitting to fully purge ordering bias
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        # Feature Scaling: Apply standard normalization to balance optimization geometry
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        print(" -> Partition mapping complete. Data normalized via StandardScaler successfully.")

    def execute_process_stage(self) -> None:
        """
        STAGE 2: PROCESS (The Proximity Principle)
        Trains the K-Nearest Neighbors pattern classifier on scaled historical arrays.
        """
        print(f"[2/3] Initiating Process Stage: Tuning KNN Model (K={self.n_neighbors})...")
        
        # Memorize map coordinates using structural fit matrices
        self.model.fit(self.X_train, self.y_train)
        print(" -> Model fitting completed. Geometric proximity maps constructed inside frame.")

    def execute_output_stage(self) -> None:
        """
        STAGE 3: OUTPUT VALIDATION
        Applies learned structural boundaries to test sets and evaluates performance vectors.
        Exposes the diagnostic tools: Confusion Matrix and Harmonic F1 Evaluation.
        """
        print("[3/3] Initiating Output Stage: Validating Model Patterns...")
        
        # Generate model classification choices based on testing features
        predictions = self.model.predict(self.X_test)
        
        print("\n====================================================")
        print("      DECODELABS SUPERVISED PATTERN EVALUATION      ")
        print("====================================================\n")
        
        # 1. Structural Diagnostic Tool: Confusion Matrix Layer
        print("--- DIAGNOSTIC CONFUSION MATRIX ---")
        matrix = confusion_matrix(self.y_test, predictions)
        matrix_df = pd.DataFrame(
            matrix, 
            index=[f"Actual {name.capitalize()}" for name in self.target_names],
            columns=[f"Predicted {name.capitalize()}" for name in self.target_names]
        )
        print(matrix_df.to_string())
        print("\n----------------------------------------------------")
        
        # 2. Precision, Sensitivity (Recall), and F1 Harmonic Mean Metrics
        print("--- COMPLETE STRATEGIC TRADE-OFF LAYOUT ---")
        report = classification_report(
            self.y_test, 
            predictions, 
            target_names=[name.capitalize() for name in self.target_names]
        )
        print(report)
        print("====================================================")

    def run_pipeline(self) -> None:
        """
        Spins up the lifecycle chain sequence of the classification engine.
        """
        self.execute_input_stage()
        self.execute_process_stage()
        self.execute_output_stage()


if __name__ == "__main__":
    # Instantiate the formal engineering object framework and run
    pipeline = IrisClassificationPipeline(n_neighbors=5, test_size=0.20)
    pipeline.run_pipeline()