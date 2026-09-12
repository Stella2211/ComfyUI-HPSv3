# ComfyUI-HPSv3

**English** | [日本語版はこちら](README.ja.md)

This ComfyUI extension evaluates images and prompts locally and generates prompts from images using NF4 models for HPSv3 and HPSv3++.

| Node | Inputs | Behavior / outputs |
| --- | --- | --- |
| HPSv3++ Model Loader | Model folder | Automatically downloads the standard model when it is missing and outputs a model configuration for the two processing nodes |
| HPSv3++ Score | Model, image, prompt | Saves and previews PNGs with scores, and outputs IMAGE and FLOAT |
| HPSv3++ Caption | Model, image | Outputs a generated prompt for each image as STRING |
| HPSv3 Model Loader | Model folder | Automatically downloads the standard HPSv3 NF4 model when it is missing and outputs a model configuration |
| HPSv3 Score | HPSv3 model, image, prompt | Saves and previews PNGs with scores, and outputs IMAGE and FLOAT |
| HPSv3 Caption | HPSv3 model, image | Outputs a generated prompt for each image as STRING |

HPSv3 and HPSv3++ models are incompatible. Connect a Model Loader from the same series to Score and Caption. Existing HPSv3++ workflows continue to work as-is.

## Requirements

- An NVIDIA CUDA GPU with BF16 support. 12 GB of VRAM is a guideline. (8 GB may work, but this is untested.)
- CPU, AMD, and Apple GPUs are not supported.
- Python 3.12 or later and Transformers 5.17.x. Manager installs the required dependencies automatically.
- An NVIDIA driver that supports ComfyUI's PyTorch and CUDA. The tested environment uses CUDA 13.0.
- The model weights require about 5.9 GB for HPSv3 and about 6.5 GB for HPSv3++. Leave enough space for the host environment dependencies and model files. Use the PyTorch and torchvision supplied by ComfyUI; they are not replaced.

## Installation

After you install the extension, Model Loader automatically downloads the standard model when the workflow runs for the first time. You do not need to run a separate command to retrieve the model.

### Install with ComfyUI-Manager

Start with ComfyUI-Manager available. If Manager is not shown, see the [official Manager installation instructions](https://docs.comfy.org/manager/install) for your ComfyUI distribution.

1. Open **Manager** in ComfyUI. In the new UI, set the search type to **Node Pack**. In the legacy UI, open the custom node list from **Install Nodes**. Labels may differ by Manager version and display language.
2. Search for `ComfyUI-HPSv3`. If it is not found, also search for `hpsv3` and remove filters such as Installed.
3. Open the result details and confirm that the repository is [`Stella2211/ComfyUI-HPSv3`](https://github.com/Stella2211/ComfyUI-HPSv3).
4. Click **Install**. If you are asked to choose a version, select a numbered version published in the Registry. In the new UI, choose it from **Version** in the details.
5. Wait for the installation to finish. The first installation adds dependencies to the host environment and may take some time. If an error appears, check the ComfyUI terminal or logs.
6. Restart ComfyUI as instructed by Manager and reload the browser if needed. Then continue to [Verify the installation](#verify-the-installation) below.

Manager retrieves the required files automatically. It connects to GitHub during installation and to Hugging Face when the model is downloaded for the first time. No additional commands are required. If installation fails, check the connection and reinstall through Manager.

For comparisons, use the same model and environment. Updating the model or dependencies may change the scores.

For an existing installation, repair its dependencies in Manager and then restart ComfyUI.

See the [new UI user guide](https://docs.comfy.org/manager/pack-management) or the [legacy UI user guide](https://docs.comfy.org/manager/legacy-ui) for UI details. Search and installation in the new UI use the Registry. Check whether a version is available on the [Registry page](https://registry.comfy.org/nodes/comfyui-hpsv3). If you cannot find it, update ComfyUI and Manager and restart them. A version being processed for publication in the Registry may not appear as an installation candidate until processing is complete.

### Install manually with Git

If you do not use Manager, run the following commands in `ComfyUI/custom_nodes` to place the extension, then install the dependencies from `requirements.txt` into ComfyUI's Python environment. If Manager has already installed it, do not clone the same extension into a second folder.

```sh
git clone https://github.com/Stella2211/ComfyUI-HPSv3.git
cd ComfyUI-HPSv3
uv pip install --python <ComfyUI Python> -r requirements.txt
```

After installing the dependencies, complete setup with the following command. This step is not required when Manager installed the extension.

```sh
uv run --no-project --python <ComfyUI Python> python install.py
```

### Download and run the HPSv3 model

Select `HPSv3-bnb-NF4` in **HPSv3 Model Loader** and connect it to **HPSv3 Score** or **HPSv3 Caption**. You can also use the [HPSv3 scoring example](examples/hpsv3_score.json). Select an image with Load Image, enter a prompt in Score's `prompt`, and run the workflow.

If it is not already present, [`stella221125/HPSv3-bnb-NF4`](https://huggingface.co/stella221125/HPSv3-bnb-NF4) is automatically downloaded to `ComfyUI/models/hpsv3/HPSv3-bnb-NF4/`. This is a merged NF4 model, so you do not need to place or convert a base model or the original reward checkpoint separately.

To place it manually, download the complete model from the Hugging Face model page and place it in `ComfyUI/models/hpsv3/HPSv3-bnb-NF4/`.

You need the complete set, including the model config, reward_config, tokenizer, and processor, rather than the weights alone. Inference uses local files only.

To evaluate an image using its generated description, load the [HPSv3 Caption→Score example](examples/hpsv3_caption_and_score.json). It passes the same image to Caption and Score and connects Caption's output to Score's `prompt`. Select an image with Load Image and run the workflow.

### Automatically download the HPSv3++ model

Select `HPSv3-PlusPlus-bnb-NF4` in Model Loader and run a workflow connected to Score or Caption. It appears in the model list even when the model is not present. On the first run, the complete model is downloaded from Hugging Face: [`stella221125/HPSv3-PlusPlus-bnb-NF4`](https://huggingface.co/stella221125/HPSv3-PlusPlus-bnb-NF4).

The default location is `ComfyUI/models/hpsv3pp/HPSv3-PlusPlus-bnb-NF4/`. The first run needs an internet connection and at least about 6.5 GB of free space; progress appears in the ComfyUI terminal or logs. The downloaded model is reused, and Score and Caption inference continue to run offline.

A new download retrieves the latest publicly available model. Existing models are not updated automatically. To retrieve an updated version, move the existing model folder elsewhere and run the workflow again.

If you cancel during download or the connection fails, check the connection and free space and run the workflow again. Partial data is retained for retry, and the model is not used until the download and verification finish. Automatic retrieval applies only to the standard model.

### Manually place a model (optional)

For an offline setup, obtain the complete model from Hugging Face and place it in the corresponding `ComfyUI/models/hpsv3/` or `ComfyUI/models/hpsv3pp/` directory. You need `config.json`, `reward_config.json`, the tokenizer, the processor, and all safetensors shards.

### Verify the installation

1. Restart ComfyUI after installing the extension.
2. Search for `HPSv3++` in the node search and confirm that the three **Model Loader**, **Score**, and **Caption** nodes appear.
3. Load the [sample workflow](examples/caption_and_score.json). Select an image with **Load Image** and select `HPSv3-PlusPlus-bnb-NF4` in **Model Loader**. No sample image is included.
4. Run the workflow and confirm that the scored image from Score is previewed and saved. The description generated by Caption is passed to Score's `prompt`. If the model is not present, it is downloaded first, so the initial run takes some time.

| Symptom | Check |
| --- | --- |
| The nodes are still missing after restart, or a loading error appears | Confirm that the extension is enabled in Manager and check the ComfyUI startup log for errors from this extension. |
| `transformers` or `bitsandbytes` cannot be found | Repair or reinstall this extension's dependencies in Manager and restart ComfyUI. Do not replace PyTorch or torchvision. |
| Automatic model download fails | Check the internet connection, access to Hugging Face, free space, and write permission for the destination, then run the workflow again. See the ComfyUI terminal or logs for details. |
| A manually placed model is reported as incomplete | Check the location and complete model set described above. Existing incomplete models are not overwritten automatically. Add the missing files, or move the existing folder elsewhere and try the standard automatic download. |
| A CUDA error or GPU memory error appears | Check the GPU and driver requirements above, close other applications using the GPU, and try again. |

## Usage

Load the [sample workflow](examples/caption_and_score.json) into ComfyUI, select an image in Load Image and a model in Model Loader, and try the connected Caption-to-Score workflow.

### Evaluate images

1. Connect Model Loader to Score's `model`.
2. Connect an IMAGE such as Load Image to Score's `images`, and enter the prompt to evaluate in `prompt`.
3. Choose `score_mode` and run the workflow.

- `banner`: Adds white space above the image and displays the score. It does not cover the original image.
- `metadata`: Does not draw text on the image. Saves the model name, score, and evaluation prompt as JSON in the PNG text item `hpsv3pp` (`hpsv3` for HPSv3).
- `both`: Displays the score in the space above the image and saves the model name, score, and evaluation prompt in the metadata of the same PNG.

You can use the same date and node-value substitutions as the standard Save Image node. For example, `%date:yyyy%/%date:MM%/%date:dd%/HPSv3pp` can save a file such as `output/2026/09/08/HPSv3pp_00001_.png`.

The metadata belongs to the PNG saved by this node. Connecting the output IMAGE to another Save Image node does not carry this score metadata over.

The same prompt can be applied to multiple images. If the prompt is a list, pass the same number of prompts as images. IMAGE and FLOAT are returned as per-image lists in input order.

### Generate prompts from images

Connect Model Loader and images to Caption. `max_new_tokens` sets the maximum generation length. Caption does not recover the original generation prompt; it generates a short description from the image. Review the content before using it.

Connect Caption's output to Score's `prompt` to evaluate the image using its generated description. For multiple images, pass the same images to both nodes in the same order.

The HPSv3 and HPSv3++ Caption→Score examples also connect a **Generated Caption** node (ComfyUI's standard Preview as Text) to display the generated description. After running the workflow, inspect the caption in this node.
