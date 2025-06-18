graph TD
    %% === Node Styling Definitions ===
    classDef kernel fill:#DDEBF7,stroke:#5B9BD5,stroke-width:2px
    classDef data fill:#E2F0D9,stroke:#70AD47,stroke-width:2px
    classDef sync fill:#FFF2CC,stroke:#FFC000,stroke-width:2px
    classDef root fill:#F2F2F2,stroke:#808080,stroke-width:2px
    classDef choice_box fill:#fafad2,stroke:#b8860b,stroke-width:1.5px,stroke-dasharray: 5 5
    classDef choice_kernel fill:#fff2cc,stroke:#d6b656,stroke-width:2px

    %% === Phase 1: Setup ===
    subgraph Phase 1: Setup and Data Upload
        direction LR
        N1[1. Input H2D]:::data
        N2[2. Targets H2D]:::data
        N3[3. Zero Gradients]:::data
    end

    %% === Phase 2 & 3: Forward Pass and Parallel Backward Pass ===
    subgraph "Phase 2 & 3: Forward Pass and Parallel Backward Pass"

        %% Forward Pass up to the choice
        N4[4. Shared Layer Forward Pass]:::kernel --> N5[5. Exits Forward Pass]:::kernel

        %% The HOST CHOICE super-node for N6
        subgraph "6. Calculate Ensemble Recipe (Weights) [HOST CHOICE]"
            style "6. Calculate Ensemble Recipe (Weights) [HOST CHOICE]" choice_box
            direction TB
            subgraph "Tier 1: Small Exits (N <= 64)"
                N6_S[ensemble_weights_reg_reduce]:::choice_kernel
            end
            subgraph "Tier 2: Medium Exits (64 < N <= 512)"
                N6_M[ensemble_weights_local_reduce]:::choice_kernel
            end
            subgraph "Tier 3: Large Exits (N > 512)"
                N6_L1[ensemble_weights_map_exp_conf]:::choice_kernel --> N6_L2[reduce_partial_sums]:::choice_kernel --> N6_L3[ensemble_weights_finalize]:::choice_kernel
            end
        end

        %% Dependencies entering the choice block
        N5 --> N6_S
        N5 --> N6_M
        N5 --> N6_L1

        subgraph "A: The Loss Calculation Path"
            direction TB
            N7["7. Blend Ensemble Probabilities"]:::kernel
            N8[8. Calculate Partial Loss]:::kernel
            N9[9. Aggregate Total Loss]:::kernel

            N7 --> N8 --> N9
            N2 --> N8
        end

        %% Dependencies exiting the choice block and forking
        N6_S --> N7
        N6_M --> N7
        N6_L3 --> N7

        subgraph "B: The Gradient Calculation Path"
            direction TB
            N10[10. Compute Exit Gradients]:::kernel
            N11["11. Aggregate & <br/> Backprop Activation"]:::kernel
            N12["12. Compute Layer Gradients <br/> (for W & b)"]:::kernel
            N13["13. Backpropagate Input Gradient <br/> (for previous layer)"]:::kernel
            N14[14. Compute Temperature Gradients]:::kernel

            N10 --> N11
            N11 --> N12
            N11 --> N13

        end

        %% Dependencies from choice block to gradients
        N6_S & N4 --> N10
        N6_M & N4 --> N10
        N6_L3 & N4 --> N10

        %% Dependencies for the temperature gradient path
        N6_S & N5 --> N14
        N6_M & N5 --> N14
        N6_L3 & N5 --> N14
    end

    %% Other dependencies
    N3 --> N10 & N12 & N13 & N14
    N2 --> N5

    %% === Phase 4: Parameter Update ===
    subgraph Phase 4: Apply Optimizer
        N15a["15. Adam Update (Shared)"]:::kernel
        N15b["15. Adam Update (Exits)"]:::kernel
        N15c["15. Adam Update (Temps)"]:::kernel
        N16[16. Clamp Temperatures]:::kernel

        N12 --> N15a
        N10 --> N15b
        N14 --> N15c
        N15c --> N16
    end

    %% === Phase 5: Synchronization ===
    subgraph Phase 5: Finalization and Monitoring
        N17[17. Copy Results to Host]:::data
        N18[18. Host Synchronization Point]:::sync
        Host[Host CPU Waits]:::root

        %% The Host layer requires specific data for monitoring and diagnostics.
        N9 --> N17   %% Total Loss
        N7 --> N17   %% Final Probabilities (for accuracy)
        N13 --> N17  %% Input Gradient (for diagnostics)

        %% The Host must wait for all parameter updates AND data transfers to complete.
        N15a & N15b & N16 --> N18
        N17 --> N18
        N18 --> Host
    end
