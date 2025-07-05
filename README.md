# Entelechy  (unstable, history will be re-written)

![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)
![Type](https://img.shields.io/badge/type-ML%20Framework-9cf)

> **Entelechy** (n.) /ɛnˈtɛləki/ — from the Aristotelian Greek *entelékheia*, the state of being complete and fully realized; the actualization of potential.

This project is an engineering philosophy made manifest. It is a research framework built on the belief that true creative velocity—the power to conduct fearless and ambitious experiments—is not born from chaos, but from a foundation of profound structural integrity.

Entelechy is a system designed to allow complex machine learning architectures to achieve their final, most efficient form.

## The Philosophy: Freedom Through Structure

At its core, Entelechy is governed by a simple but powerful idea: a strict separation of concerns that creates a federation of autonomous architectures.

*   **A Constitution, Not a Dictatorship:** The framework itself is a minimalist conductor. It provides a universal "constitution" that governs how its member architectures must interact with the system, but it never dictates their internal affairs. [You can read this constitution here (Repository Structure: Philosophy and Specification)](./STRUCTURE.md).

*   **Sovereign Architectures:** Each ML architecture is a sovereign entity. It manages its own code, its own tests, and its own scientific goals. It is free to be as simple or as complex as it needs to be, as long as it honors the federal contract.

*   **The Build is a Promise:** The framework guarantees that if an experiment can be built, it can be run. By treating the compile step as a sacred validation gate, Entelechy eliminates entire classes of runtime errors, freeing the researcher to focus on science, not syntax.

## An Organism of Pure Dataflow
[Averaging Ensembled Classifier: A Computational Organism](./architectures/averaging_ensembled_classifier/README.md)

To see the power of this philosophy, consider one of its first children: a **Unified, Memory-Aware Streaming Engine** designed for complex multi-head classifiers.

This architecture is not merely a collection of functions; it is a living system that adapts to its environment.

*   It is composed of simple, single-purpose cells (**Modular Kernels**) which are brilliantly orchestrated by a central nervous system (**The Host Orchestrator**).

*   It follows a single, unified "always stream" dataflow, allowing it to gracefully handle problems of any scale—from a problem so tiny it fits in registers, to one that requires chunking a "data tsunami" across terabytes of input.

*   Most profoundly, when faced with immense memory pressure, the architecture makes a strategic choice between speed and survival. It can **cache** intermediate results in memory for maximum performance, or it can **recompute** them on the fly to guarantee scalability, trading time for space in a beautiful act of self-preservation.

This is what Entelechy makes possible: architectures that are not just powerful, but intelligent and resilient.

## Getting Started

The developer contract is designed to be simple and predictable.

1.  **Provision:** Install the required tools.
    ```bash
    pip install -r requirements.txt
    ```
2.  **Configure:** Let the framework verify your environment.
    ```bash
    meson setup build
    ```
3.  **Validate & Build:** Compile all code and generate all runnable experiments. A successful exit is a guarantee of runtime integrity.
    ```bash
    meson compile -C build
    ```
4.  **Execute:** Run a specific, declaratively-defined experiment.
    ```bash
    meson run -C build <architecture_name>/run-<network_name>
    ```

## License

Entelechy is licensed under the **GNU Affero General Public License v3.0**.

This license is chosen to foster an open and collaborative research community. It ensures that any modifications or applications that are made available over a network must also have their source code shared under the same terms. We believe this is the strongest guarantee that the work will remain free, open, and for the benefit of all.