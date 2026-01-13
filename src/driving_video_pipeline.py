import kagglehub
import pandas as pd
import os
from dataset.video_io import DrivingVideoMiniDataset
from video_learner import learn_batch_data, evaluate_batch_data
from models.perceptor import Perceptor
def load_dataset():
    """Load the dataset_mini as train, val, test splits."""
    
    # Create train/val/test datasets
    datasets = DrivingVideoMiniDataset.create_dataloaders(
        dataset_dir="dataset_mini",
        train_ratio=0.7,
        val_ratio=0.2,
        random_seed=42
    )
    
    print(f"\n=== DATASET LOADED ===")
    print(f"Train videos: {len(datasets['train'])}")
    print(f"Val videos: {len(datasets['val'])}")  
    print(f"Test videos: {len(datasets['test'])}")
    
    return datasets 

def initialize_model():
  perceptor = Perceptor()
  
  object_state_extractor = None 
  events_abstractor = None 
  rule_reasoner = None 
  knowledge_reasoner = None
  model = {
      "perceptor": perceptor,
      "object_state_extractor": object_state_extractor,
      "events_abstractor": events_abstractor, 
      "rule_reasoner": rule_reasoner,
      "knowledge_reasoner": knowledge_reasoner
  }
  return model



def main():
    # Load the dataset with train/val/test splits
    datasets = load_dataset()
    model = initialize_model()
    
    train_dataset = datasets["train"]
    val_dataset = datasets["val"]
    test_dataset = datasets["test"]
    
    print(f"\n=== ITERATIVE LEARNING PROCESS ===")
    print(f"Will learn from {len(train_dataset)} training videos iteratively")
    
    knowledge = {}
    # Iterative learning: learn from one video, then validate, then test
    for idx in range(len(train_dataset)):
        print(f"\n--- ITERATION {idx + 1}/{len(train_dataset)} ---")
        
        # Learn from one training video
        print(f"Learning from training video {idx + 1}")
        train_batch = train_dataset[idx]  # Each video is a batch
        knowledge = learn_batch_data(model, train_batch, knowledge)
        
        # Validate on all validation data
        print(f"Validating after training video {idx + 1}")
        for val_idx in range(len(val_dataset)):
            val_batch = val_dataset[val_idx]
            evaluate_batch_data(model, val_batch, knowledge)
        
        # Test on all test data
        print(f"Testing after training video {idx + 1}")
        for test_idx in range(len(test_dataset)):
            test_batch = test_dataset[test_idx]
            evaluate_batch_data(model, test_batch)
    
    return 
  

if __name__ == "__main__":
    
    main()