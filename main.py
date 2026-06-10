import argparse
from src.loader import load_config, load_data
from src.features import scale_features
from src.model import train_model, evaluate_model, get_feature_importance
from src.visualizer import plot_feature_importance, plot_confusion_matrix, plot_label_distribution

def get_inputs():
    parser = argparse.ArgumentParser(description="Exoplanet Classifier Pipeline")
    parser.add_argument("--data", required=True, help="Path to dataset CSV")
    parser.add_argument("--config", required=True, help="Path to config JSON")
    return parser.parse_args()

def show_menu():
    print("\n=============================")
    print("   Exoplanet Classifier Menu  ")
    print("=============================")
    print("1. Load and clean data")
    print("2. Train model")
    print("3. Evaluate model")
    print("4. Generate plots")
    print("5. Run full pipeline")
    print("0. Exit")
    print("=============================")
    return input("Choose an option: ").strip()

def main():
    args = get_inputs()
    
    config = None
    df = None
    X, y, scaler = None, None, None
    model, X_test, y_test = None, None, None

    while True:
        choice = show_menu()

        if choice == "1":
            print("\nLoading config and data...")
            config = load_config(args.config)
            df = load_data(args.data, config)
            print(f"Done. Dataset shape: {df.shape}")

        elif choice == "2":
            if df is None:
                print("\nLoad data first (option 1).")
                continue
            print("\nScaling features and training model...")
            X, y, scaler = scale_features(df, config)
            model, X_test, y_test = train_model(X, y)
            print("Done. Model trained.")

        elif choice == "3":
            if model is None:
                print("\nTrain model first (option 2).")
                continue
            print("\nEvaluating model...")
            evaluate_model(model, X_test, y_test)
            get_feature_importance(model, config)

        elif choice == "4":
            if model is None:
                print("\nTrain model first (option 2).")
                continue
            print("\nGenerating plots...")
            plot_feature_importance(model, config)
            plot_confusion_matrix(model, X_test, y_test)
            plot_label_distribution(df, config)

        elif choice == "5":
            print("\nRunning full pipeline...")
            config = load_config(args.config)
            df = load_data(args.data, config)
            print(f"Dataset shape: {df.shape}")
            X, y, scaler = scale_features(df, config)
            model, X_test, y_test = train_model(X, y)
            evaluate_model(model, X_test, y_test)
            get_feature_importance(model, config)
            plot_feature_importance(model, config)
            plot_confusion_matrix(model, X_test, y_test)
            plot_label_distribution(df, config)
            print("Done.")

        elif choice == "0":
            print("\nExiting. Goodbye.")
            break

        else:
            print("\nInvalid option. Try again.")

if __name__ == "__main__":
    main()