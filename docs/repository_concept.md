### **Framework Design Philosophy and Specification**

#### **Preamble**

This document specifies the governing philosophy and mandated structure of the repository. It serves as the authoritative reference for the project's architecture. The document's own structure is designed to be hierarchical, flowing from a high-level objective to core design principles, and finally to a concrete implementation specification. This ensures that the underlying philosophy is clearly delineated from the specific tooling and that the rationale for each implementation choice is traceable.

---

### **1. Primary Objective**

**The primary objective of this framework is to facilitate efficient, reproducible, and reliable experimentation.**

This objective informs all architectural decisions. Any proposed change to the framework should be evaluated based on its impact on the process of developing, testing, and executing experiments.

---

### **2. Core Design Principles**

The primary objective is supported by the following four design principles.

1.  **Hierarchical Design:** The framework is structured as a composition of clearly defined abstractions, not as a monolithic system. This conceptual hierarchy must be explicitly defined and consistently reflected in the repository's physical directory structure to aid in comprehensibility and maintain a logical separation of concerns.

2.  **Architectural Autonomy:** Each machine learning architecture is treated as an autonomous, self-contained module. It manages its own source code, tests, and experiment definitions. The framework provides a lightweight federation layer, defining a common interface for building and executing these autonomous units, rather than dictating their internal implementation.

3.  **Declarative Interaction:** User interaction with the framework is intended to be declarative. Developers define the desired state through manifests (e.g., build files), specifying *what* should be configured and built. The framework is responsible for executing the necessary imperative steps to achieve that state, thereby abstracting away procedural complexity.

4.  **Build-Time System Guarantees:** A strict separation must be maintained between the development and runtime environments. The build process functions as a mandatory validation gate. A successful build provides a formal guarantee of syntactic correctness and complete dependency resolution, thereby ensuring the integrity of the runtime environment.

---

### **3. Specification of Implementation**

The following specifications define the concrete implementation required to realize the Core Principles.

#### **3.1. The Federated Directory Structure**
To implement the principles of *Hierarchical Design* and *Architectural Autonomy*, the repository must adhere to a tiered file system structure.

*   **Level 1: Framework Root (`/`)**: Contains the top-level manifest for orchestrating the collection of architectures.
*   **Level 2: Architectures Collection (`/architectures/`)**: The parent directory for all autonomous architecture modules.
*   **Level 3: Autonomous Architecture (`/architectures/<architecture_name>/`)**: A self-contained module directory, including its own source, assets, and manifest.

```
ml_engine_framework/
├── architectures/
│   └── <architecture_name>/
│       └── meson.build           # Architecture Manifest
└── meson.build                     # Framework Conductor Manifest
```

#### **3.2. The Tiered Manifest System**
To support *Declarative Interaction* and provide *Build-Time System Guarantees*, the build system is defined by tiered manifests.

*   **The Conductor Manifest (`/meson.build`):** A minimal manifest whose responsibility is to discover and delegate build operations to each sub-directory within `/architectures/`.
*   **The Architecture Manifest (`/architectures/<architecture_name>/meson.build`):** A comprehensive manifest declaring the architecture's internal components, dependencies, and a list of all provided runnable experiments.

#### **3.3. The Decoupled Runtime Bridge**
To enforce the separation of concerns between build and runtime environments, execution is handled through configured launchers.

1.  A generic launcher template file is used as a blueprint.
2.  During compilation, the build system populates this template with experiment-specific parameters drawn from an Architecture Manifest.
3.  This process generates a final, configured, and directly executable script. The core runtime code remains unaware of this configuration process.

---

### **4. Developer Workflow**

The practical workflow resulting from the preceding specifications is as follows.

1.  **Provisioning (Developer):** Install required toolchains and packages.
    *   `pip install -r requirements.txt`
2.  **Configuration (Framework):** The build system is configured and the environment is verified.
    *   `meson setup build`
3.  **Validation & Generation (Framework):** The framework validates all code and generates all runtime artifacts.
    *   `meson compile -C build`
4.  **Execution (Developer):** A specific, declaratively-defined experiment is invoked.
    *   `meson run -C build <architecture_name>/run-<network_name>`