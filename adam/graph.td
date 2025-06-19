graph TD
    %% === Node Styling Definitions ===
    classDef kernel fill:#DDEBF7,stroke:#5B9BD5,stroke-width:2px
    classDef data fill:#E2F0D9,stroke:#70AD47,stroke-width:2px
    classDef sync fill:#FFF2CC,stroke:#FFC000,stroke-width:2px
    classDef root fill:#F2F2F2,stroke:#808080,stroke-width:2px
    classDef choice_box fill:#fafad2,stroke:#b8860b,stroke-width:1.5px,stroke-dasharray: 5 5
    classDef choice_kernel fill:#fff2cc,stroke:#d6b656,stroke-width:2px
    classDef path_a fill:#e2f0d9,stroke:#70ad47,stroke-dasharray: 5 5
    classDef path_b fill:#ddebf7,stroke:#5b9bd5,stroke-dasharray: 5 5
    classDef note fill:#fefefe,stroke:#ccc,stroke-width:1px,stroke-dasharray:2 2
    classDef hierarchical_tier fill:#e6e6fa,stroke:#9370db,stroke-width:2px

    %% === Phase 1: Setup & Pre-computation ===
    subgraph Phase 1: Setup and Data Upload
        direction LR
        N1[1. Input H2D]:::data
        N2[2. Targets H2D]:::data
        N3[3. Zero Gradients]:::data
    end

    %% === Phase 2: Common Forward Pass Trunk ===
    subgraph "Phase 2: Common Forward Pass Trunk"
        N4[4. Shared Layer Forward Pass]:::kernel --> N5[5. Exits Forward Pass]:::kernel
        subgraph "6. Calculate Ensemble Recipe [HOST CHOICE]"
            style "6. Calculate Ensemble Recipe [HOST CHOICE]" choice_box
            direction TB

            N6_S["<b>Tier 1: Register Reduce</b><br/><i>(≤ 64 exits)</i>"]:::choice_kernel
            N6_M["<b>Tier 2: Local Memory Reduce</b><br/><i>(65 to 100K exits)</i>"]:::choice_kernel

            subgraph "Tier 3: Hierarchical DAG Reduction (> 100K exits)"
                direction TB
                style "Tier 3: Hierarchical DAG Reduction (> 100K exits)" fill:#f5f0ff, stroke:#9370db, stroke-width:1px
                N6_H1["6c.1 Process Chunks<br/><i>(Kernel Dispatched in Parallel)</i>"]:::hierarchical_tier
                N6_H2["6c.2 Reduce Chunk Pairs<br/><i>(Looped Kernel Chain)</i>"]:::hierarchical_tier
                N6_H3["6c.3 Final Normalization<br/><i>(Single Kernel)</i>"]:::hierarchical_tier
                N6_H1 --> N6_H2 --> N6_H3
            end
        end

        %% Input to the ensemble calculation
        N5 --> N6_S & N6_M & N6_H1
        N2 --> N5
    end

    %% === FORK POINT ===
    N7["7. Blend Ensemble Probabilities"]:::kernel

    %% Outputs from all possible tiers feed into the Blend kernel
    N6_S --> N7
    N6_M --> N7
    N6_H3 --> N7

    %% === Phase 3 & 4: Parallel Execution Paths ===
    subgraph "Phase 3 & 4: Parallel Execution Paths"
        direction TB

        subgraph "Path A: High-Priority Inference Stream"
            style "Path A: High-Priority Inference Stream" path_a
            N7_D2H["8a. D2H Early Results <br/>(Final Probabilities)"]:::data
        end

        subgraph "Path B: Lower-Priority Training Pipeline"
            style "Path B: Lower-Priority Training Pipeline" path_b
            subgraph "Loss Calculation"
                N8_B[8b. Calculate Partial Loss]:::kernel --> N9_B[9b. Aggregate Total Loss]:::kernel
            end

            subgraph "Gradient Calculation"
                N6_S & N6_M & N6_H3 & N4 & N5 --> N10_B
                N10_B[10b. Compute Exit Gradients]:::kernel --> N11_B["11b. Aggregate & Backprop"]:::kernel
                N11_B --> N12_B["12b. Layer Gradients"]:::kernel & N13_B["13b. Input Gradient"]:::kernel
                N14_B[14b. Temperature Gradients]:::kernel
            end

            subgraph "Parameter Update"
                N15a_B["15b. Adam Update (Shared)"]:::kernel
                N15b_B["15b. Adam Update (Exits)"]:::kernel
                N15c_B["15b. Adam Update (Temps)"]:::kernel
                N14_B --> N15c_B --> N16_B[16b. Clamp Temperatures]:::kernel
                N12_B --> N15a_B
                N10_B --> N15b_B
            end
        end

        %% Dependencies connecting the common trunk to the parallel paths
        N7 --> N7_D2H
        N7 & N2 --> N8_B

        %% Ensemble results from ALL possible tiers are required for backprop
        N6_S & N6_M & N6_H3 & N4 --> N10_B
        N6_S & N6_M & N6_H3 & N5 --> N14_B
        N3 --> N10_B & N12_B & N13_B & N14_B
    end

    note_hierarchical["**Dynamic DAG Construction:**<br/>For Tier 3, the host dynamically builds a<br/>Merkle-style reduction tree of kernels.<br/>The 'Reduce Chunk Pairs' step is a chain of kernels<br/>looped until a single result remains."]:::note
    N6_H2 --. note_hierarchical

    %% === Phase 5: Asynchronous Synchronization & Host Interaction ===
    subgraph "Phase 5: Asynchronous Synchronization & Host Interaction"
        direction TB

        subgraph "Host Interaction Sequence"
           style "Host Interaction Sequence" fill:#f2f2f2, stroke-width:0px
           Host_Action1["Host Processes Early Result<br/>(non-blocking)"]:::root
           Host_Action2["Host Waits for Full Batch Completion<br/>(blocking before next iteration)"]:::root
           Host_Action1 --> Host_Action2
        end

        Sync_A["8c. Inference Stream Sync"]:::sync
        Sync_B["18. Final Batch Sync"]:::sync

        Sync_A --> Host_Action1
        Sync_B --> Host_Action2

        N7_D2H --> Sync_A

        N16_B --> Sync_B
        subgraph "17b. D2H Late Diagnostics"
            N17_Diag[ ]:::data
        end
        N9_B --> N17_Diag
        N13_B --> N17_Diag
        N17_Diag --> Sync_B
    end
