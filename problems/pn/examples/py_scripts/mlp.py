import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import argparse
import sys

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a simple neural network for digit classification')
    parser.add_argument('train_csv', help='Path to the training CSV file')
    parser.add_argument('test_csv', help='Path to the test CSV file')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs (default: 10)')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate (default: 0.01)')
    parser.add_argument('--inputs', type=int, default=784, help='Number of input features (default: 784)')
    parser.add_argument('--hidden', type=int, default=16, help='Hidden layer size (default: 16)')
    
    args = parser.parse_args()
    
    # Load data
    try:
        train_df = pd.read_csv(args.train_csv)
        test_df = pd.read_csv(args.test_csv)
    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading CSV files: {e}")
        sys.exit(1)
    
    print(f"Loaded training data: {train_df.shape}")
    print(f"Loaded test data: {test_df.shape}")
    
    # Prepare data
    X = train_df.drop('label', axis=1).values / 255.0
    y = train_df['label'].values
    
    # Check if test data has the same number of features as training data
    expected_features = X.shape[1]
    print(f"Expected number of features: {expected_features}")
    
    # Handle test data - remove any extra columns that might be indices or IDs
    if test_df.shape[1] > expected_features:
        # If test data has more columns, assume the first column is an index/ID
        print(f"Test data has {test_df.shape[1]} columns, removing first column (assumed to be index)")
        X_test = test_df.iloc[:, 1:].values / 255.0
    else:
        X_test = test_df.values / 255.0
    
    print(f"Test data shape after processing: {X_test.shape}")
    
    # Verify dimensions
    if X_test.shape[1] != expected_features:
        print(f"Error: Test data has {X_test.shape[1]} features, but training data has {expected_features} features")
        sys.exit(1)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.LongTensor(y_val)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    class MLP(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(MLP, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, num_classes)
        
        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x
    
    model = MLP(args.inputs, args.hidden, 10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    print(f"Training MLP with {args.inputs} inputs, {args.hidden} hidden neurons for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        model.train()
        for data, labels in train_loader:
            outputs = model(data)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, labels in val_loader:
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        print('Epoch {}: {:.2f}%'.format(epoch+1, 100 * correct / total))

if __name__ == "__main__":
    main() 