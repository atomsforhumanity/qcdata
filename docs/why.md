# Why qcdata?

Other data structure package for quantum chemistry exist such as [QCElemental](https://github.com/MolSSI/QCElemental) and [Atomic Simulation Environment](https://wiki.fysik.dtu.dk/ase/index.html). I often found these libraries were too heavy weight or too feature poor for most of what I needed. I also found their APIs to be somewhat unintuitive and often cumbersome to use. `qcdata` is designed to be easy to use, easy to reason about, and provide a unified format for diverse quantum chemistry calculations.

`qcdata` is also the central package used by [`qccompute`](https://github.com/atomsforhumanity/qccompute) to power interoperable quantum chemistry calculations across dozens of packages and [`bigchem`](https://github.com/mtzgroup/bigchem), which scales QC calculations across hundreds of nodes on academic clusters or the cloud. I needed robust and reliable data structures for powering these applications. `qcdata` is the result of thousands of hours of organizing quantum chemistry data into coherent data structures.

It is my hope that you'll find `qcdata` to be intuitive and delightful to use. If you have any questions please open an [issue](https://github.com/atomsforhumanity/qcdata/issues) on GitHub. Pull requests welcome :)

## Getting Started

Check out the [API Documentation](./api/overview.md) to understand how qcdata works. Take a look at the [Visualizations](./visualizations/overview.md) to see how easy it is to view your results using qcdata!
