import pandas as pd

def preprocess_data(file_path, output_path):
    data = pd.read_csv(file_path)
    
    # Handle missing values
    data.fillna(data.median(), inplace=True)

    # Encoding categorical variables
    data = pd.get_dummies(data, drop_first=True)

    # Save preprocessed data
    data.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")

# Example Usage
if __name__ == "__main__":
    preprocess_data('../data/raw_data.csv', '../data/preprocessed_data.csv')
