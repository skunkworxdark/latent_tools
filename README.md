# `latent_tools` for InvokeAI (v4.0+)

 A collection of experimental custom nodes for the InvokeAI image generation platform (version 4.0 and above). These nodes provide capabilities for manipulating, analyzing, and visualizing the latent space within InvokeAI's workflows. These should all work with both Flux, SD1.5, SD2, SDXL and SD3.5.

Discord Link: [`latent_tools`](https://discord.com/channels/1020123559063990373/1352361385060860025) for support, discussion, and feedback.

## Nodes

- `Latent Average` - adds both latents together and divides by 2 - Channels can be selected with all or a 0-based csv list
- `Latent Combine` - Many more combination methods. Quite a few of these are AI generated so Not sure how useful they will be but might be interesting. + Channel Selection.
- `Latent Plot` - generates a grid of histograms one per channel of the latent. Idea stolen from @dunkeroni previous posts. lots of options + Channel Selection.
- `Latent Channels to Grid` - a grid of Gray-scale images that represent the channels in a latent.
- `Latent Dtype Convert` -  convert a latent to Float61 Float32 or BFloat16
- `Latent Modify` - allow scale and shift of latents channel vales.
- `Latent Match` - MATCHING_METHODS  "histogram", "std_dev", "mean", "std_dev+mean", "cdf", "moment", "range", "std_dev+mean+range" + Channel Selection.
- `Latent White Noise` -  Outputs white noise for Flux, SD3.5 or SD/SDXL
- `Latent Low Pass filter` - apply a Low pass filter to a latent (brown noise ~0.05)
- `Latent Band Pass filter` - apply a Band pass filter to a latent (green noise ~ 0.1-0.2)
- `Latent High Pass filter` - apply a High pass filter to a latent (blue noise ~0.15)
- `Latent Blend Linear` - Blend two latents in a linear fashion
- `Latent Normalize Std-Dev` - Normalize a latent to a specified Std Dev
- `Latent Normalize Range` - Normalize a latent to a specific Range
- `Latent Normalize STD Dev and Range` -  imperatively attempts to normalize a latent to a Std-Dev and Range at the same time. 


## Usage

#### <ins>Install</ins><BR>
There are two options to install the nodes:

1. **Recommended**: Git clone into the `invokeai/nodes` directory. This allows updating via `git pull`.

    - Open your terminal or command prompt and navigate to your InvokeAI "nodes" folder.
    - Run the following command::
    ```bash
    git clone https://github.com/skunkworxdark/latent_tools.git
    ```

2. Manually download [latent_tools.py](latent_tools.py) & [__init__.py](__init__.py) then place them in a subfolder (e.g., `latent_tools`) under `invokeai/nodes`. 

#### <ins>Update</ins><BR>
Run a `git pull` from the `latent_tools` folder.

Or run the provided `update.bat`(windows) or `update.sh`(Linux).

For manual installs, download and replace the files.

#### <ins>Remove</ins><BR>
Delete the `latent_tools` folder. Or rename it to `_latent_tools`` so InvokeAI will ignore it.

## ToDo
- better readme
- Add more useful latent tools ....

# Example Usage
ToDo