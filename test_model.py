import torch
from models.teacher_model.model import TeacherEvaluator

def test_teacher_evaluator():
    model = TeacherEvaluator()
    batch_size, seq_len, input_size = 2, 10, 768
    x = torch.randn(batch_size, seq_len, input_size)
    try:
        outputs = model(x)
        print("Success! Outputs:", outputs)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_teacher_evaluator()