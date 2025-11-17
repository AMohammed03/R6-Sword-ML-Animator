import torch
import json
import numpy as np
from model import AnimationGenerator, convert_flat_to_nested, KEYFRAME_TIMES

def load_trained_model(model_path='trained_model.pth'):
    checkpoint = torch.load(model_path)

    model = AnimationGenerator(
        input_size=2,
        output_size=checkpoint['num_features'] * checkpoint['max_keyframes'],
        hidden_sizes=checkpoint['config']['hidden_sizes'],
        dropout=checkpoint['config']['dropout_rate']
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, checkpoint

def generate_new_animation(pattern, output_file='generated_animation.json'):
    model, checkpoint = load_trained_model()

    with torch.no_grad():
        pattern_tensor = torch.FloatTensor([pattern])
        prediction = model(pattern_tensor).numpy()[0]

        keyframes = prediction.reshape(
            checkpoint['max_keyframes'],
            checkpoint['num_features']
        )

        time_column = np.array(KEYFRAME_TIMES).reshape(-1, 1)
        keyframes_with_time = np.concatenate([time_column, keyframes], axis=1)

        nested_keyframes = [
            convert_flat_to_nested(kf, checkpoint['feature_cols'])
            for kf in keyframes_with_time
        ]

        animation = {
            'pattern': f"{pattern[0]}-{pattern[1]}",
            'keyframes': nested_keyframes
        }

        with open(output_file, 'w') as f:
            json.dump(animation, f, indent=2)

        return animation

if __name__ == "__main__":
    test_pattern = [6, 4]
    generate_new_animation(test_pattern)