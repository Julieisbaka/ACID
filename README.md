# ACID

ACID (Attention Conservation under Input Degradation) is a AI benchmark designed to measure how models perform under diffrent context lengths compared to their baseline performance. It evaluates the model's ability to retain reasoning capabilities as the input context becomes increasingly noisy or bloated.

## Noise

In ACID, "noise" refers to extraneous or irrelevant information added to the input context. The benchmark tests how well a model can maintain its reasoning performance when the input is saturated with such noise. Noise is introduced in discrete tiers, ranging from minimal to extremely large context lengths, to simulate real-world scenarios where models must process vast amounts of information while still focusing on the relevant parts of the input.

