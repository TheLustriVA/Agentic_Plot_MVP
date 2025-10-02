# Agentic_Plot_MVP

A minimal test implementation of AI agents for creative writing

## Aim for this project

To create a SmolAgents agentic tool powered locally by llama.cpp that takes a basic Markdown file with sections for brief descriptions of a setting, one or more characters, a scenario, and at least two potential endings, and uses the AI agents to expand those inputs into:

- A cohesive plot based on the archetypes described by Christopher Booker in "The Seven Basic Plots: Why We Tell Stories" and the given scenario
- Clearly-defined, fleshed-out, and setting-appropriate character descriptions
- A map of how the plot might get to each of the provided endings
- A 1000-word synopsis for each of those plots

## Software setup

### Code

Python 3.12 using UV for package management.

### Inference engine

Llama.cpp's `llama-server` API serving a local 24B-32B model that can best get the job done.

### Agentic framework

Huggingface's [SmolAgents framework](https://github.com/huggingface/smolagents) for simplicity.

### Basic IO

Markdown files for both input and output with Typer used as the python CLI library.

#### Inputs

A single markdown file with the following sections:

- A brief description of the setting
- Brief descriptions of required characters and how they relate to the setting and scenario
- A description of the scenario, that is, the events the characters face
- Brief descriptions of at least two potential endings given the above

#### Outputs

Separate markdown files for:

- information about the plot archetype chosen by the agentic AI and a beat-by-beat description of the plot for each ending
- descriptions of all characters required for the plot to work, from brief mentions of incidental characters to extended descriptions of user-mandated characters
- a world-building bible for the elements of the setting required for the plot and characters to work
- a debrief with technical details

## Hardware setup

### System

- Host: snowblind Kernel: 6.14.0-32-generic arch: x86_64 bits: 64
- Desktop: GNOME v: 48.0 Distro: Ubuntu 25.04 (Plucky Puffin)

### Machine

- Type: Desktop System: ASUS product: N/A v: N/A serial: ::superuser required::
- Mobo: ASUSTeK model: ROG STRIX X870-A GAMING WIFI v: Rev 1.xx
  - serial: ::superuser:: required> UEFI: American Megatrends v: 0505
  - date: 09/27/2024

### CPU

- Info: 12-core AMD Ryzen 9 9900X3D [MT MCP] speed (MHz): avg: 5361
  - min/max: 600/5575
  
### Graphics

- Device-1: NVIDIA GB202 [GeForce RTX 5090] driver: nvidia v: 580.65.06
- Display: wayland server: X.Org v: 24.1.6 with: Xwayland v: 24.1.6
  - compositor: gnome-shell driver: gpu: nv_platform,nvidia,nvidia-nvswitch
  - resolution: 3840x2160~60Hz
- API: OpenGL v: 4.6.0 compat-v: 4.5 vendor: nvidia mesa v: 580.65.06
  - renderer: NVIDIA GeForce RTX 5090/PCIe/SSE2
- Info: Tools: api: eglinfo,glxinfo gpu: nvidia-settings,nvidia-smi
  - x11: xdriinfo, xdpyinfo, xprop, xrandr
  
### Network

- Device-1: Intel Ethernet I226-V driver: igc
- Device-2: MEDIATEK driver: mt7925e

### Drives

- Local Storage: total: 4.55 TiB used: 3.6 TiB (79.1%)

### Info

- Memory: total: 192 GiB available: 184.08 GiB used: 40.39 GiB (21.9%)
- Processes: 972 Uptime: 1d 9h 55m Shell: Zsh inxi: 3.3.37
