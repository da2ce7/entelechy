graph TD
    %% === Node Styling Definitions ===
    classDef kernel fill:#DDEBF7,stroke:#5B9BD5,stroke-width:2px
    classDef data fill:#E2F0D9,stroke:#70AD47,stroke-width:2px
    classDef sync fill:#FFF2CC,stroke:#FFC000,stroke-width:2px
    classDef root fill:#F2F2F2,stroke:#808080,stroke-width:2px
    classDef choice_box fill:#fff8dc,stroke:#daa520,stroke-width:1.5px,stroke-dasharray: 5 5
    classDef choice_kernel fill:#fff2cc,stroke:#d6b656,stroke-width:2px
    classDef path_a fill:#e2f0d9,stroke:#70ad47,stroke-dasharray: 5 5
    classDef path_b fill:#ddebf7,stroke:#5b9bd5,stroke-dasharray: 5 5
    classDef note fill:#fefefe,stroke:#ccc,stroke-width:1px,stroke-dasharray:2 2
    classDef hierarchical_tier fill:#e6e6fa,stroke:#9370db,stroke-width:2px

    %% === Graph Title & Explanatory Note ===
    note_problem_type["**Host-Level Dispatch:**<br/>The host (Python) selects a `PROBLEM_TYPE` (e.g., CCE or BCE).<br/>This single choice determines which kernel is executed inside each<br/>of the 'HOST CHOICE' boxes, ensuring mathematical consistency<br/>throughout the training pipeline."]:::note

    %% === Phase 1: Setup & Pre-computation ===
    subgraph Phase 1: Setup and Data Upload
        direction LR
        N1[1. Input H2D]:::data
        N2[2. Targets H2D]:::data
        N3[3. Zero Gradients]:::data
    end

    %% === Phase 2: Common Forward Pass Trunk with Dynamic Loss ===
    subgraph "Phase 2: Common Forward Pass Trunk"
        N4[4. Shared Layer Forward Pass]:::kernel

        subgraph "5. Exits Fwd Pass & Loss [HOST CHOICE]"
            style "5. Exits Fwd Pass & Loss [HOST CHOICE]" choice_box
            N5_CCE["<b>compute_all_exits_cce</b><br/><i>(Softmax + CCE Loss)</i>"]:::choice_kernel
            N5_BCE["<b>compute_all_exits_bce</b><br/><i>(Sigmoid + BCE Loss)</i>"]:::choice_kernel
        end

        subgraph "6. Calculate Ensemble Recipe [HOST CHOICE]"
            style "6. Calculate Ensemble Recipe [HOST CHOICE]" choice_box
            direction TB
            N6_S["<b>Tier 1: Register Reduce</b><br/><i>(≤ 64 exits)</i>"]:::choice_kernel
            N6_M["<b>Tier 2: Local Memory Reduce</b><br/><i>(65 to 100K exits)</i>"]:::choice_kernel
            subgraph "Tier 3: Hierarchical DAG Reduction (> 100K exits)"
                direction TB
                style "Tier 3: Hierarchical DAG Reduction (> 100K exits)" fill:#f5f0ff, stroke:#9370db, stroke-width:1px
                N6_H1["6c.1 Process Chunks"]:::hierarchical_tier --> N6_H2["6c.2 Reduce Chunk Pairs"]:::hierarchical_tier --> N6_H3["6c.3 Final Normalization"]:::hierarchical_tier
            end
        end

        %% Dependencies for Phase 2
        N4 --> N5_CCE & N5_BCE
        N2 --> N5_CCE & N5_BCE
        N5_CCE & N5_BCE --> N6_S & N6_M & N6_H1
    end

    %% === FORK POINT ===
    N7["7. Blend Ensemble Probabilities"]:::kernel

    %% Outputs from choice boxes feed into the Blend kernel
    N6_S & N6_M & N6_H3 --> N7
    N5_CCE & N5_BCE --> N7

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
                subgraph "10b. Compute Exit Gradients [HOST CHOICE]"
                    style "10b. Compute Exit Gradients [HOST CHOICE]" choice_box
                    N10_CCE["<b>grads_exit_cce</b><br/><i>(CCE Derivative)</i>"]:::choice_kernel
                    N10_BCE["<b>grads_exit_bce</b><br/><i>(BCE Derivative)</i>"]:::choice_kernel
                end

                N11_B["11b. Aggregate & Backprop"]:::kernel
                N10_CCE & N10_BCE --> N11_B

                N11_B --> N12_B["12b. Layer Gradients"]:::kernel & N13_B["13b. Input Gradient"]:::kernel

                subgraph "14b. Temperature Gradients [HOST CHOICE]"
                    style "14b. Temperature Gradients [HOST CHOICE]" choice_box
                    N14_CCE["calculate_temp_gradients_cce"]:::choice_kernel
                    N14_BCE["calculate_temp_gradients_bce"]:::choice_kernel
                end
            end

            subgraph "Parameter Update"
                N15a_B["15b. Adam Update (Shared)"]:::kernel
                N15b_B["15b. Adam Update (Exits)"]:::kernel
                N15c_B["15b. Adam Update (Temps)"]:::kernel
                N14_CCE & N14_BCE --> N15c_B --> N16_B[16b. Clamp Temperatures]:::kernel
                N12_B --> N15a_B
                N10_CCE & N10_BCE --> N15b_B
            end
        end

        %% Dependencies connecting the common trunk to the parallel paths
        N7 --> N7_D2H
        N7 & N2 --> N8_B

        %% Dependencies for Gradient Calculations (Path B)
        %% Consistent path dependencies: CCE path depends on CCE outputs, BCE depends on BCE outputs.
        N5_CCE --> N10_CCE & N14_CCE
        N5_BCE --> N10_BCE & N14_BCE

        %% Common dependencies for the gradient choice boxes
        N4 & N6_S & N6_M & N6_H3 --> N10_CCE & N10_BCE & N14_CCE & N14_BCE
        N3 --> N10_CCE & N10_BCE & N12_B & N13_B & N14_CCE & N14_BCE
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
