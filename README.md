# Model 1: 4-Parameter (Single Exponential Rise with Linear Drift)

$$I(t) = I_{offset} + A \cdot \left(1- e^{-t/\tau}\right) + D \cdot t$$

This is a phenomenological model used when you believe the resealing/uptake can be described by **one dominant kinetic process** ($\tau$), but you need to correct for measurement artifacts like photobleaching. 

- $I_{offset}$ (Baseline Intensity): The initial fluorescence background intensity at $t=0$.

- $A$ (Amplitude): The total magnitude of the fluorescence increase due to dye entry after the pulse.

- $tau$ (Single Time Constant): Represents the average characteristic time of the combined dye influx/pore resealing process. It treats the resealing as a single-step event.

- $D$ (Linear Drift Term): This term accounts for any slow, steady change in intensity over time, like photobleaching (if $D < 0$) or slow focus drift. It is an artifact correction, not a physical resealing process.

# Model 2: 5-Parameter (Double Exponential Resealing)

$$I(t) = A_{f} - A_{1} \ e^{-t/\tau_{1}} - A_{2} \ e^{-t/\tau_{1}}$$

This is the standard model in electroporation literature (like the paper you cited) because it accounts for the **multi-stage nature of pore resealing**, which is rarely a single event.

- $A_{f}$ (Final Intensity): The steady-state (plateau) intensity reached when the process is complete.

- $A_{1}$ (Fast Amplitude): The fraction of the total signal associated with the fast resealing/uptake component.

- $tau_{1}$ (Fast Time Constant): Represents the time scale for the quick process, typically related to the rapid closure of small, numerous pores.

- $A_{2}$ (Slow Amplitude): The fraction of the total signal associated with the slow resealing/uptake component.

- $tau_{2}$ (Slow Time Constant): Represents the time scale for the slow process, often related to the reorganization or closure of large, stable pores (macropores) or membrane repair.

# Key Conceptual Difference

The choice between the two models depends on what you hypothesize drives the long-term changes in your signal:
- **Choose Model 1 (4-PARAM) if**: You believe the dye uptake is kinetically simple (one $\tau$), and the long-term drift you see in your plot is likely a linear measurement artifact (e.g., photobleaching). You use the $D \cdot t$ term to strip away the artifact and find the true $\tau$ of the primary event.

- **Choose Model 2 (5-PARAM) if**: You believe the dye uptake and subsequent membrane resealing is inherently complex and multi-staged. The $\tau_1$ and $\tau_2$ values provide distinct physical insights into the different types of pores or repair mechanisms involved in the resealing process. If there is also photobleaching, you would typically correct the raw data for linear drift before applying this model.
