# Douglas 3D Splat

Small learning package for implementing 3D Gaussian Splatting from scratch.

Install from the repository root:

```bash
pip install -e douglas_3dsplat
```

Optional reference backend:

```bash
pip install -e "douglas_3dsplat[ref]"
```

On the current Modal image used by this repo, prefer the matching prebuilt
`gsplat` wheel if you want faster setup:

```bash
pip install ninja numpy jaxtyping rich
pip install gsplat --index-url https://docs.gsplat.studio/whl/pt22cu121
```
