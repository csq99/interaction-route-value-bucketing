# Interaction-Aware One-Shot DeiT Router

This isolated experiment replaces independent layer contribution labels with route-level task utility.

- First, a `DynamicAdapter` is trained with randomly sampled legal routes, covering multiple skip transitions.
- Then, for each training batch, several newly sampled routes are executed and labelled by `CE + logit-KD` loss.
- A one-shot router reads layer-0 CLS, patch-token mean, and patch-token standard deviation. It emits unary layer values and pairwise interaction values.
- At inference, the router scores all legal middle-layer subsets. For DeiT 2x, this is `C(10,4)=210` scalar route scores and requires no additional DeiT forwards.

The default Slurm script is a pilot: 2,048 training images, 512 evaluation images, one coverage-adapter epoch, and one router epoch. It is a feasibility test, not a final ImageNet result.
