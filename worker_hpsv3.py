import json
from pathlib import Path
import sys
import warnings


sys.path.insert(0, str(Path(__file__).resolve().parent / "third_party" / "hpsv3-4bit" / "hpsv3" / "src"))

from evaluation.hpsv3_quantized import HPSv3QuantizedInferencer


def run(request):
    scorer = HPSv3QuantizedInferencer.from_merged_dir(
        merged_dir=request["model"], processor_dir=request["model"],
        device=request["device"], local_files_only=True,
    )
    if request["operation"] == "score":
        return [scorer.score([image], [prompt])[0]
                for image, prompt in zip(request["images"], request["prompts"], strict=True)]
    if request["operation"] == "caption":
        return scorer.caption(request["images"], max_new_tokens=request["max_new_tokens"])
    raise ValueError("Unknown HPSv3 operation")


def main():
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = run(request)
    messages = list(dict.fromkeys(str(item.message) for item in caught))
    payload = {"result": result, "warnings": messages}
    Path(sys.argv[2]).write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
