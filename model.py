import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim

CONFIG = {
    'data_file': "formatted_animation_data.json",
    "output_file": "generated_animation.json",
    "num_epochs": 250,
    "learning_rate": 0.001,
    "test_size": 0.2,
    "random_state": 42,
    "hidden_sizes": [128, 256, 256],
    "dropout_rate": 0.2
}
FEATURE_JOINTS = {
    "HumanoidRootPart": ["Rotation", "Position"],
    "Torso": ["Rotation", "Position"],
    "Left Arm": ["Rotation", "Position"],
    "Right Arm": ["Rotation", "Position"],
    "Head": ["Rotation", "Position"],
    "Handle": ["Rotation", "Position"]
}
KEYFRAME_TIMES = [0, 0.35, 0.40, 0.45, 0.67]



def load_json_data(file_path):
    data = {}

    try:
        with open(file_path, 'r') as f:
            data = json.load(f) 
    except FileNotFoundError:
        print(f"Error: Could not find the file '{file_path}'.")
        exit()
    except Exception as e:
        print(f"An error occurred: {e}")
        exit()

    return data

def extract_poses_recursive(pose_dict, row_data):
    for joint_name, joint_data in pose_dict.items():
        if joint_name in FEATURE_JOINTS:
            properties = FEATURE_JOINTS[joint_name]
            
            for prop_name in properties:
                if prop_name in joint_data:
                    coords = joint_data[prop_name]
                    row_data[f'{joint_name}_{prop_name}_X'] = coords[0]
                    row_data[f'{joint_name}_{prop_name}_Y'] = coords[1]
                    row_data[f'{joint_name}_{prop_name}_Z'] = coords[2]
        
        if isinstance(joint_data, dict):
            for key, value in joint_data.items():
                if isinstance(value, dict) and key not in ['Rotation', 'Position']:
                    extract_poses_recursive({key: value}, row_data)

def parse_animations(data):
    all_animations = []
    animation_counter = 0

    for pattern, animations_list in data.items():
        # Extract pattern numbers
        pattern_1 = int(pattern[0])
        pattern_2 = int(pattern[2])
        
        # Loop through each animation
        for animation in animations_list:
            animation_id = f"{pattern}_{animation_counter}"
            animation_counter += 1
            
            keyframes_data = animation['keyframes']
            
            # Store all keyframes for this animation
            animation_keyframes = []
            
            for kf in keyframes_data:
                kf_data = {}
                kf_data['Time'] = kf['Time']
                extract_poses_recursive(kf['Poses'], kf_data)
                animation_keyframes.append(kf_data)
            
            # Store this complete animation
            all_animations.append({
                'animation_id': animation_id,
                'pattern_1': pattern_1,
                'pattern_2': pattern_2,
                'keyframes': animation_keyframes
            })

    return all_animations

def prepare_training_data(all_animations):
    X = []  # Will store [pattern_1, pattern_2]
    y = []  # Will store flattened keyframe sequences

    # Get feature names from first animation
    feature_cols = [k for k in all_animations[0]['keyframes'][0].keys() if k != "Time"]
    num_features = len(feature_cols)
    max_keyframes = max(len(anim['keyframes']) for anim in all_animations)

    for anim in all_animations:
        # Input: pattern
        X.append([anim['pattern_1'], anim['pattern_2']])
        
        # Output: flatten all keyframes
        keyframe_sequence = []
        for kf in anim['keyframes']:
            # Extract values in consistent order
            kf_values = [kf[col] for col in feature_cols]
            keyframe_sequence.extend(kf_values)
        
        y.append(keyframe_sequence)

    return np.array(X),  np.array(y), feature_cols, num_features, max_keyframes

class AnimationGenerator(nn.Module):
    def __init__(self, input_size=2, output_size=None, hidden_sizes=[128, 256, 256], dropout=0.2):
        super(AnimationGenerator, self).__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size

        layers = layers[:-1]
        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def train_model(model, X_train, y_train, config):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])  # Added parentheses

    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)

    for epoch in range(config["num_epochs"]):
        model.train()
        
        predictions = model(X_train_tensor)
        loss = criterion(predictions, y_train_tensor)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{config["num_epochs"]}], Loss: {loss.item():.4f}')

    return loss.item()

def evaluate_model(model, X_test, y_test):
    criterion = nn.MSELoss()
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)

    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor)
        test_loss = criterion(predictions, y_test_tensor)

    return test_loss.item(), predictions.numpy()

def convert_flat_to_nested(flat_keyframe, feature_cols):
    nested = {
        "Time": float(flat_keyframe[0]),
        "Poses": {}
    }

    for i, col_name in enumerate(feature_cols):
        parts = col_name.rsplit("_", 2)

        if len(parts) == 3:
            joint_name = parts[0]
            prop_type = parts[1]
            axis = parts[2]

            nestedJointParent = nested["Poses"]

            if joint_name == "Torso":
                nestedJointParent = nestedJointParent["HumanoidRootPart"]
            elif joint_name != "HumanoidRootPart":
                nestedJointParent = nestedJointParent["HumanoidRootPart"]["Torso"]

            if joint_name == "Handle":
                nestedJointParent = nestedJointParent["Right Arm"]


            nestedJoint = None
            if not joint_name in nestedJointParent:
                nestedJointParent[joint_name] = {}
            
            nestedJoint = nestedJointParent[joint_name]
            if not prop_type in nestedJoint:
                nestedJoint[prop_type] = []

            nestedJoint[prop_type].append(float(flat_keyframe[i + 1]))
    return nested

def generate_animation(model, pattern, feature_cols, num_features, max_keyframes):
    model.eval()

    with torch.no_grad():
        pattern_tensor = torch.FloatTensor([pattern])
        prediction = model(pattern_tensor).numpy()[0]

        keyframes = prediction.reshape(max_keyframes, num_features)
        time_column = np.array(KEYFRAME_TIMES).reshape(-1, 1)
        keyframes_with_time = np.concatenate([time_column, keyframes], axis=1)

        nested_keyframes = [
            convert_flat_to_nested(kf, feature_cols)
            for kf in keyframes_with_time
        ]

        return {
            "pattern": f"{pattern[0]}-{pattern[1]}",
            "keyframes": nested_keyframes
        }

def save_animation(animation, file_path):
    with open(file_path, "w") as f:
        json.dump(animation, f, indent=2)

def main():
    raw_data = load_json_data(CONFIG['data_file'])
    all_animations = parse_animations(raw_data)

    X, y, feature_cols, num_features, max_keyframes = prepare_training_data(all_animations)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=CONFIG['test_size'],
        random_state=CONFIG['random_state'],
        stratify=X[:, 0]
    )

    model = AnimationGenerator(
        input_size=2,
        output_size=y.shape[1],
        hidden_sizes=CONFIG['hidden_sizes'],
        dropout=CONFIG['dropout_rate']
    )

    final_train_lost = train_model(model, X_train, y_train, CONFIG)

    torch.save({
        'model_state_dict': model.state_dict(),
        'feature_cols': feature_cols,
        'num_features': num_features,
        'max_keyframes': max_keyframes,
        'config': CONFIG,
    }, 'trained_model.pth')
    
    test_loss, predictions = evaluate_model(model, X_test, y_test)

    test_pattern = [6, 4]
    generated = generate_animation(model, test_pattern, feature_cols, num_features, max_keyframes)
    save_animation(generated, CONFIG['output_file'])



if __name__ == "__main__":
    main()


