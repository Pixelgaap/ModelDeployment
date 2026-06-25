import argparse
from transformers import pipeline
from PIL import Image


def load_pipeline(model_name_or_path, device=0):
    return pipeline(task="image-to-text", model=model_name_or_path, device=device)


def infer(pipe, image_path, max_new_tokens=512):
    image = Image.open(image_path).convert("RGB")
    return pipe(image, max_new_tokens=max_new_tokens)


def main():
    parser = argparse.ArgumentParser(description="Nanonets-OCR2-3B 模型推理脚本")
    parser.add_argument("--model", type=str, required=True, help="模型路径或名称")
    parser.add_argument("--device", type=int, default=0, help="使用的GPU设备索引，-1表示CPU")
    parser.add_argument("--image", type=str, required=True, help="输入图像路径")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="生成文本的最大长度")

    args = parser.parse_args()

    print(f"加载模型: {args.model}")
    pipe = load_pipeline(args.model, device=args.device)

    print(f"处理图像: {args.image}")
    outputs = infer(pipe, args.image, max_new_tokens=args.max_new_tokens)

    print("\n=== 推理结果 ===")
    for i, output in enumerate(outputs):
        print(f"输出 {i+1}:")
        if isinstance(output, dict) and "generated_text" in output:
            print(f"  识别文本: {output['generated_text']}")
        else:
            print(f"  {output}")


if __name__ == "__main__":
    main()