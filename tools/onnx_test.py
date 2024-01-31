import onnxruntime as ort
import torch


def main():
    x = torch.randn(1, 3, 448, 448)
    providers = ['CUDAExecutionProvider']

    ort_sess = ort.InferenceSession('model.onnx', providers=providers, )
    outputs = ort_sess.run(None, {'input': x.numpy()}, )
    print(outputs)
    pass


if __name__ == "__main__":
    main()

