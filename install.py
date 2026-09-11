"""Manager installation hook for the separately stored HPSv3++ source.

Manager installs requirements.txt first. This hook only obtains the reviewed
upstream files; it does not install packages or download model weights.
"""


def main():
    from _vendor.hpsv3_4bit.hpsv3pp.upstream import ensure_source

    directory = ensure_source()
    print(f"HPSv3++ upstream source verified: {directory}")


if __name__ == "__main__":
    main()
