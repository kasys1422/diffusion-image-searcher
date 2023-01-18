[[日本語](https://github.com/kasys1422/diffusion-image-searcher/tree/main)/English]
# diffusion-image-searcher

diffusion-image-searcher is software that searches your local computer for images related to the text you enter. It can also search for similar images based on the image.

## How to install

### windows exe version (recommended)

1. download the latest version of the file from within Release and unzip it to any location.
2. (Optional) Download the additional models from within Release and move the unzipped file into /res/model.
3. double-click the executable file in the extracted folder to run it.

(Note) Anti-virus software may detect false positives, so please specify exceptions or take other measures as appropriate.

### python version

1. Install python3.9 .
2. Create a folder in an arbitrary location and clone the repository there.
3. Download the latest version of the files from within Release and copy the trained models from /res/model to the appropriate hierarchy.
4. Create and activate the python virtual environment.
5. install the modules based on requirements.txt.
6. run diffusion_image_searcher.py.

(Note) Since the operation has been confirmed on Windows, errors may occur when executing on other operating systems.

## How to use

### Search for images from text
1. Start the software.
2. (Optional, if you have already downloaded additional models and have at least 12 GB of free memory) Specify "stable-diffusion-v1-4-openvino-fp16-CPU" for "Image Generation Model" in the settings.
3. Select the "Search from text" tab.
4. Specify the "Folder to search".
5. Describe the characteristics of the image you want to search for in English in the "Prompt" field.
6. Click the "Search" button.
7. Wait for a while as the search starts. (It may take more than 10 minutes depending on the number of images searched and disk access speed.)
8. The results will be listed.

### Search for similar images by image
1. Start the software.
2. Select the "Search by image" tab.
3. Select the "Folder to search
4. Select an image from "File to search" that is close to the image you want to search.
5. Click the "Search" button.
6. Wait for a while for the search to start. (It may take more than 10 minutes depending on the number of images searched and disk access speed.)
7. The results will be listed.

## How it works

Based on the input text, the search is achieved by generating an image using a diffusion model and calculating the similarity between the image and the file in the local computer.

## System requirements

<details>
  <summary>
    windows exe version
  </summary>
  <dl>
    <dt>OS</dt>
    <dd>Windows 10 or Windows 11/dd>
    <dt>CPU</dt>
    <dd>x64 CPU with 4 or more cores supporting AVX2 or SSE2 instructions (Intel, 2019 or later recommended) <br>* CPU supporting AVX or SSE2 instructions</dd>
    <dt>RAM</dt>
    <dd>16GB or more *12GB or more</dd>
    <dt>ROM</dt>
    <dd>10 GB free space or more</dd>
    <dt>Display</dt>
    <dd>Wider display area than 1280x720 resolution at 100% magnification</dd>
    *Minimum operating requirements
  </dl>
</details>

<details>
  <summary>
    python version
  </summary>
  <dl>
    <dt>Python Version</dt>
    <dd>python 3.9</dd>
    <dt>CPU</dt>
    <dd>x64 CPU with 4 or more cores supporting AVX2 or SSE2 instructions (Intel, 2019 or later recommended) <br>* CPU supporting AVX or SSE2 instructions</dd>
    <dt>RAM</dt>
    <dd>16GB or more *12GB or more</dd>
    <dt>ROM</dt>
    <dd>10 GB free space or more</dd>
    <dt>Display</dt>
    <dd>Wider display area than 1280x720 resolution at 100% magnification</dd>
    *Minimum operating requirements
  </dl>
</details>
