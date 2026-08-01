# Dataset sources and public scene names

The release does not redistribute datasets or pretrained models. Users should
obtain the data from the official sources and follow their terms. The paper
uses the following public scene labels:

| Scene labels in the paper | Dataset family | Public source |
| --- | --- | --- |
| Lego, Hotdog | NeRF Synthetic | [NeRF project](https://www.mildenhall.net/nerf/) |
| Train, Truck | Tanks and Temples | [Tanks and Temples benchmark](https://www.tanksandtemples.org/download/) |
| DrJohnson, Playroom | Deep Blending | [Deep Blending publication](https://discovery.ucl.ac.uk/id/eprint/10117776/) |
| Bicycle, Garden | Mip-NeRF 360 | [Mip-NeRF 360 project](https://jonbarron.info/mipnerf360/) |

`Train` is the paper-facing name for the historical internal `tt107k` label.
The internal label is retained only in private experiment manifests and is not
required by the public CLI. A user-side run supplies a model adapter, dataset
root, camera list, and a profile; the resulting manifest should record those
inputs without embedding private absolute paths.
