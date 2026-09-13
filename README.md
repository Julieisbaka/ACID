# ACID

ACID (Attention Conservation under Input Degradation) is a AI benchmark designed to measure how models perform under diffrent context lengths compared to their baseline performance. It evaluates the model's ability to retain reasoning capabilities as the input context becomes increasingly noisy or bloated.

## Noise

In ACID, "noise" refers to extraneous or irrelevant information added to the input context. The benchmark tests how well a model can maintain its reasoning performance when the input is saturated with such noise. Noise is introduced in discrete tiers, ranging from minimal to extremely large context lengths, to simulate real-world scenarios where models must process vast amounts of information while still focusing on the relevant parts of the input.

Noise consists of wikipedia articles. Certain precussions are taken to ensure that such articles do not hijack the benchmark. These precussions include the stripping of questions from the end of articles to prevent prompt hijacking as well as the removal of any content that could directly answer benchmark questions.

It is important to note that while noise is derived from real-world sources, it is carefully curated to avoid introducing bias or unfair advantages to any particular model. The goal is to create a challenging yet controlled environment to accurately assess a model's robustness to input degradation.

## Scoring

Scoring in ACID, takes the average of runs at different noise levels and compares it to the baseline performance of the model without any added noise. This allows for a quantifiable measure of how well the model maintains its reasoning capabilities as the input context becomes increasingly degraded.

### Subscores

The scores are farther broken down into domain specific subscores, allowing for a more granular analysis of a model's performance across different types of tasks or content areas. This helps identify strengths and weaknesses in specific domains, providing deeper insights into the model's capabilities and limitations.

The current subscore domains consist of:

ACID-Math: Mathematics and physics.
ACID-Bio: Biology and life sciences.
ACID-Code: Programming and software development.
ACID-Chem: Chemistry.

## Goals

While ACID does want to accurately reflect real-world usecases that is not the goal of the benchmark, instead its goal is to measure how models perform under varying levels of input degradation. The benchmark aims to quantify the model's robustness and ability to maintain reasoning performance as the context becomes increasingly noisy or bloated.

Ultimately, ACID seeks to provide insights into the limitations and strengths of different models when faced with challenging input conditions, helping researchers and developers design more resilient AI systems.