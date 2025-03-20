# `latent_tools` for InvokeAI (v4.0+)
Discord Link :- [latent_tools](https://discord.com/channels/1020123559063990373/1216759622007001139)

A set of InvokeAI nodes to manipulate and visualize latents in Invoke AI workflows.

Note: These are currently very experimental, especially the generic versions.

- `Latent Average` - adds both latents together and divides by 2
- `Latent Combine` - Many more combination methods. Not sure how useful these are but might give some interesting results.
- `Latent histograms` - generates a grid of histograms one per channel of the latent.
- `Latent channels to grid` - a grid of gray-scale images that represent the channels in a latent.


## Usage
### <ins>Install</ins><BR>
There are two options to install the nodes:

1. **Recommended**: Git clone into the `invokeai/nodes` directory. This allows updating via `git pull`.

    - In the InvokeAI nodes folder, run:
    ```bash
    git clone https://github.com/skunkworxdark/latent_tools.git
    ```

2. Manually download [latent_tools.py](latent_tools.py) & [__init__.py](__init__.py) then place them in a subfolder under `invokeai/nodes`. 

### <ins>Update</ins><BR>
Run a `git pull` from the `latent_tools` folder.

Or run `update.bat`(windows) or `update.sh`(Linux).

For manual installs, download and replace the files.

### <ins>Remove</ins><BR>
Delete the `latent_tools` folder. Or rename it to `_latent_tools`` so InvokeAI will ignore it.

## Useful Notes

These should all work with bot flux and SD, SDXL.  I have not tested SD3.5 yet

## ToDo
- better readme
- Add more useful latent tools ....

# Example Usage
todo
