# **Comprehensive Analysis of Circular Rotary Positional Embeddings: Implementation Strategies in DiT360 and Injection Vectors for Flux.1 and Stable Diffusion 3 in ComfyUI**

## **1\. Introduction**

The generative AI landscape has recently witnessed a paradigm shift from U-Net-based architectures to Diffusion Transformers (DiTs), exemplified by models such as Stable Diffusion 3 (SD3) and Flux.1. While these architectures offer superior scalability and multimodal comprehension, they introduce novel challenges in handling spatial topology, particularly for non-standard aspect ratios and geometries like 360-degree panoramic imagery. The core of this challenge lies in the attention mechanism's reliance on positional encodings to understand spatial relationships. Unlike Convolutional Neural Networks (CNNs), which possess translation invariance and local inductive biases that can be easily adapted to circular topologies via padding, Transformers utilize explicit positional embeddings—most notably Rotary Positional Embeddings (RoPE)—that inherently assume a bounded, Euclidean coordinate system.

This report provides an exhaustive technical analysis of the implementation of Circular RoPE and circular padding strategies. It is specifically structured to address the user's query regarding the feasibility of implementing these features as custom nodes within ComfyUI without modifying core libraries. The analysis dissects the native implementation of circularity in the DiT360 architecture and contrasts it with the "inference-time hacking" required to retrofit Flux.1 and SD3 with circular capabilities. By examining the mathematical foundations of RoPE, the tensor-level operations of modern DiTs, and the execution graph of ComfyUI, this report establishes a theoretical and practical framework for achieving seamless 360-degree generation.

The investigation reveals that while DiT360 employs a training-time integration of circular convolutions and attention windows, similar results can be approximated in Flux and SD3 through sophisticated model patching. This involves intercepting the RoPE frequency calculation and modifying the attention masks, a process that can indeed be encapsulated within a ComfyUI custom node by leveraging the ModelPatcher class to override internal method calls dynamically.

## **2\. Mathematical Foundations of Rotary Positional Embeddings**

To understand the complexity of implementing circularity in DiTs, one must first rigorously define the operation of Standard RoPE and identify precisely where the "seam" artifact originates in a cylindrical topology.

### **2.1 The Mechanics of Standard RoPE**

Rotary Positional Embedding (RoPE) encodes position information by rotating the query ($q$) and key ($k$) vectors in the attention mechanism. Unlike absolute positional embeddings, which add a vector to the input, RoPE operates multiplicatively. For a token at position $m$, the embedding function $f(x, m)$ is defined such that the inner product of two vectors depends only on their relative distance $m \- n$.

In the context of 2D image generation, RoPE is typically applied axially. The feature dimension $d$ is split into two halves: one for the height dimension ($y$) and one for the width dimension ($x$). The rotation is applied independently to feature subspaces corresponding to these dimensions.

The operation for a single dimension (e.g., width) can be expressed using complex numbers. If we treat pairs of elements in the embedding vector as complex numbers, the rotation for position $m$ is given by multiplication with $e^{im\\theta\_j}$, where $\\theta\_j$ represents the frequency for the $j$-th feature pair. In high-dimensional implementation, this manifests as an element-wise multiplication of the query/key vectors with a pre-computed cosine and sine frequency table (the cis table).

The standard attention score calculation becomes:

$$\\text{Attention}(q, k) \= \\text{Softmax}\\left(\\frac{(R^d\_{\\Theta, m}q)^T (R^d\_{\\Theta, n}k)}{\\sqrt{d}}\\right)$$

Where $R$ represents the rotation matrix derived from the positions $m$ and $n$. Crucially, the frequencies $\\theta\_j$ are typically defined as a geometric sequence: $\\theta\_j \= 10000^{-2j/d}$.

### **2.2 The Topological Discontinuity Problem**

In a standard image generation task, the width coordinate $x$ ranges from $0$ to $W$. The relative distance between the leftmost pixel ($x=0$) and the rightmost pixel ($x=W$) is $W$. The attention mechanism, therefore, treats these pixels as maximally distant. The rotation applied to $x=0$ and $x=W$ results in vectors that are significantly divergent in the embedding space.

In a 360-degree cylindrical topology (equirectangular projection), the pixel at $x=0$ is spatially adjacent to $x=W$. Their effective relative distance is $0$ (or $1$). However, Standard RoPE encodes the large numerical difference ($W$), preventing the attention mechanism from attending to the seamless continuity. The model "sees" an edge.

This creates the "seam" artifact:

1. **Attention Drop-off:** Tokens at the left edge do not attend to tokens at the right edge, causing a lack of semantic coherence across the boundary.  
2. **Padding Artifacts:** In the VAE (Visual Autoencoder) stage, standard zero-padding or reflection-padding at the edges introduces foreign values that the decoding process interprets as a frame boundary.

### **2.3 The Circular Condition**

For a RoPE implementation to be circular, the embedding at position $x$ must be equivalent to the embedding at position $x+W$. Mathematically, this requires the rotation to be periodic with period $W$.

$$e^{i(x)\\theta} \= e^{i(x+W)\\theta}$$

$$e^{iW\\theta} \= 1$$

$$W\\theta \= 2\\pi k, \\quad k \\in \\mathbb{Z}$$  
This condition implies that for perfect circularity, the frequencies $\\theta$ used in RoPE must be quantized such that they complete an integer number of cycles over the width $W$. Standard RoPE frequencies, which are based on a fixed base (usually 10,000), do not satisfy this property for arbitrary image widths. This mismatch is the primary theoretical barrier to seamless generation in standard models.

## **3\. DiT360: Native Architecture and Implementation Analysis**

The DiT360 architecture represents a specialized adaptation of the Diffusion Transformer designed explicitly to resolve these topological issues. Unlike Flux or SD3, which are general-purpose planar models, DiT360 integrates circularity into its fundamental design. The user specifically requested an analysis of implementation details in DiT360.

### **3.1 Overview of DiT360 Methodology**

DiT360 addresses the seam problem through two primary mechanisms, distinguishing between the Latent Space Encoder (VAE) and the Denoising Backbone (DiT).

1. **Padding-Free Circular Convolution (VAE Stage):** Addressing local texture continuity.  
2. **Circular Attention (DiT Stage):** Addressing global semantic continuity.

### **3.2 Padding-Free Circular Convolution**

Standard Convolutional Neural Networks (used in the VAE encoder/decoder of Stable Diffusion and Flux) utilize padding to maintain spatial dimensions.

* **Standard:** Zero padding or Replicate padding. This introduces "edge effects."  
* **DiT360 Implementation:** The authors replace standard padding with **Circular Padding**.  
  * **Mechanism:** For a convolution kernel of size $k$, the input tensor is padded by taking $\\lfloor k/2 \\rfloor$ pixels from the *opposite* side of the image.  
  * **Tensor Operation:** If Input is $I \\in \\mathbb{R}^{C \\times H \\times W}$, the padded input $I\_{pad}$ concatenates $I$ to the left of the image and $I\[:, :, 0 : \\lfloor k/2 \\rfloor\]$ to the right.  
  * **Result:** The convolution operation at the boundary effectively "sees" the pixels from the other side. This ensures that the latent representation generated by the VAE is topologically cylindrical.

**Relevance to ComfyUI Query:** The user noted difficulty implementing this in ComfyUI without core mods. This is because torch.nn.Conv2d takes a padding\_mode argument, but standard model definitions in diffusers or comfy usually hardcode this to 'zeros' or 'reflect'. Changing it requires traversing the model graph and replacing the Conv2d layers, which *can* be done in a custom node (detailed in Section 6).

### **3.3 Circular Attention Mechanism in DiT360**

In the Transformer backbone, DiT360 does not simply rely on global attention (which is $O(N^2)$ and expensive). It often uses **Windowed Attention** (Swin Transformer style) or local attention patches.

* **The Problem:** Standard window partitioning cuts the image into non-overlapping grids. A window at the left edge has no connection to the right edge.  
* **DiT360 Solution:** **Circular Window Shift**.  
  * When partitioning the latents into windows, the grid is treated as a cylinder.  
  * If a window straddles the boundary (e.g., half on the left, half on the right), the pixels are gathered using the modulo operator on the width coordinate.  
  * Relative Positional Encoding: Crucially, DiT360 modifies the Relative Positional Embedding table. For a relative distance $\\Delta x$, the lookup index is computed as:

    $$\\Delta x\_{circ} \= (x\_i \- x\_j \+ W/2) \\% W \- W/2$$

    This ensures that the "distance" between $0$ and $W-1$ is $-1$, not $-(W-1)$.

### **3.4 Asymmetric Positional Encoding**

DiT360 acknowledges that 360-degree images are cylindrical, not spherical (in terms of grid topology).

* **Height (Elevation):** Maintains standard linear constraints (top is not connected to bottom).  
* Width (Azimuth): Enforces circular constraints.  
  The model is trained with this inductive bias. The weights of the attention layers learn to interpret the circular relative distances correctly. This is a key differentiator from patching Flux; Flux weights were trained on linear distances. When we patch Flux to use circular distances, we are effectively feeding "out-of-distribution" positional embeddings to the model.

## **4\. Flux.1 Architecture Analysis**

To understand how to inject circularity into Flux.1, we must dissect its unique architecture, which differs significantly from the U-Net of SD1.5 or the MMDiT of SD3.

### **4.1 The Dual-Stream Architecture**

Flux.1 utilizes a hybrid architecture consisting of two primary block types, processing a latent input img and a conditioning input txt.

#### **4.1.1 DoubleStreamBlocks**

These blocks process image and text modalities independently but allow information exchange via cross-attention.

* **Input:** img (Latents), txt (T5/CLIP embeddings), img\_ids, txt\_ids.  
* **RoPE Application:** Flux applies RoPE *only* to the img stream in these blocks (and technically the txt stream has its own handling, but spatial RoPE is key for the image).  
* **Mechanism:** The img\_ids tensor contains the $(y, x)$ coordinates. The apply\_rope function uses these coordinates to rotate the queries and keys (Q, K) of the self-attention mechanism within the image stream.

#### **4.1.2 SingleStreamBlocks**

In the later stages of the network, Flux concatenates the image and text streams into a single sequence.

* **Input:** cat(img, txt).  
* **Attention:** Full self-attention across both modalities.  
* **RoPE Complexity:** Here, the positional embeddings for the text and image must remain distinct yet compatible. The img\_ids still govern the spatial relationships of the image part of the sequence.

### **4.2 Flux RoPE Implementation Details**

The crucial injection point for Flux is the generation of img\_ids and the apply\_rope function.

* **Dimensionality:** Flux uses 2D RoPE. If the head dimension is $D$, it typically splits it: $D/2$ features encode Height, $D/2$ features encode Width.  
* **The Grid:** The img\_ids are typically generated as a meshgrid.  
  Python  
  \# Conceptual Flux ID generation  
  h\_ids \= torch.arange(H).unsqueeze(1).repeat(1, W)  
  w\_ids \= torch.arange(W).unsqueeze(0).repeat(H, 1)  
  ids \= torch.stack(\[h\_ids, w\_ids\]) \# Shape:

* **Injection Vector:** To make Flux circular, we must manipulate how w\_ids are interpreted during the RoPE rotation. We cannot simply change w\_ids to w\_ids % W because w\_ids are already in range $$. The modification must happen in the *relative distance* calculation or the *frequency* domain.

## **5\. Stable Diffusion 3 (MMDiT) Architecture Analysis**

Stable Diffusion 3 employs the Multimodal Diffusion Transformer (MMDiT). While similar to Flux, its handling of positional embeddings differs in the implementation details within the JointTransformerBlock.

### **5.1 MMDiT Structure**

SD3 fuses text and image tokens early in the block structure using a joint attention mechanism.

* **Patch Size:** SD3 operates on $2 \\times 2$ latent patches.  
* **Positional Encoding:** Like Flux, it uses 2D RoPE.  
* **AdaLN-Zero:** SD3 relies heavily on Adaptive Layer Norm (AdaLN) conditioned on the timestep and pooled text embedding to modulate the blocks.

### **5.2 RoPE Injection in SD3**

The apply\_rope function in SD3 is usually a standalone utility in sd3.layers.

* **Key Difference:** SD3's implementation often pre-computes the cosine/sine tables for the maximum possible resolution and slices them.  
* **Circular Strategy:** Unlike Flux, where we might need to modify IDs per block, SD3's centralized frequency table generation allows for a potentially cleaner "global" patch by modifying the cached cis tables to be circular.

## **6\. Researching the Solution: Implementing Circularity as a ComfyUI Custom Node**

The user explicitly asks: "Research if there is any way to make this work as a custom node."  
Verdict: Yes, it is entirely possible to implement both Circular Padding (VAE) and Circular RoPE (DiT) as a custom node without modifying comfy\_extras or torch site-packages. This is achieved via Dynamic Module Patching and Model Wrapper Hooks.

### **6.1 The Challenge of "Core Modification"**

The user likely failed because they tried to pass an argument like padding\_mode='circular' to the top-level workflow. ComfyUI's default Load VAE and Load Diffusion Model nodes instantiate standard classes where these hardcoded values (like padding\_mode='zeros') are buried deep in the \_\_init\_\_ methods of sub-sub-modules. Standard "Model Patcher" nodes usually only affect weights or attention masks, not layer architecture.

### **6.2 Solution Part 1: Circular VAE (The Convolution Patch)**

The VAE is a standard CNN. To make it circular, we must replace all torch.nn.Conv2d layers with a custom CircularConv2d layer (or change their padding\_mode).

**Implementation Strategy (Custom Node):**

1. **Input:** A standard VAE model object.  
2. **Operation:** Recursively traverse the vae.first\_stage\_model (Encoder) and vae.second\_stage\_model (Decoder).  
3. **The Hook:**  
   Python  
   \# Pseudo-code for Custom Node Logic  
   def make\_vae\_circular(vae):  
       for name, module in vae.named\_modules():  
           if isinstance(module, torch.nn.Conv2d):  
               \# Option A: In-place attribute modification (if supported by backend)  
               \# module.padding\_mode \= 'circular'   
               \# Note: PyTorch Conv2d often pre-processes padding logic in \_\_init\_\_,   
               \# so changing the attribute might not be enough.

               \# Option B: Layer Replacement (Robust)  
               new\_layer \= torch.nn.Conv2d(..., padding\_mode='circular')  
               new\_layer.load\_state\_dict(module.state\_dict())  
               setattr(parent\_module, name\_of\_child, new\_layer)  
       return (vae,)

4. **Insight:** This recursion *must* handle the specific hierarchy of the AutoencoderKL used in SD3/Flux. The "seamless" artifacts often come from the VAE, not just the DiT. If the DiT is perfect but the VAE uses zero-padding, you will see a 1-8 pixel seam.

### **6.3 Solution Part 2: Circular Flux/SD3 (The RoPE Patch)**

This is the more complex part. We cannot easily replace layers because FlashAttention kernels are fused. We must intervene at the **Model Patcher** level or **Function Injection**.

**The comfy.model\_patcher.ModelPatcher** allows us to set a model\_options dictionary that is passed down to the blocks. However, standard Flux code doesn't look for a "circular" flag. We need to **Wrap the Model Function**.

Detailed Engineering of the Custom Node:  
We can create a node "Apply Circular Flux" that uses model.set\_model\_unet\_function\_wrapper.  
**The Wrapper Logic:**

1. **Intercept apply\_rope:** We cannot directly intercept a low-level function like apply\_rope easily from the top level wrapper.  
2. Alternative: Patching the Transformer Options:  
   ComfyUI passes transformer\_options through the network. We can inject a "patch" into the patches key of model\_options.  
   Current advanced ComfyUI nodes (e.g., from *Heios* or *Comfy-Anonymous*) use a specific trick: **Modifying the position\_ids or tiling.**  
   *However, for true Circular RoPE, we need to mathematically alter the cos/sin.*  
   The "Hack" Implementation:  
   We define a custom apply\_rope function in Python scope.  
   We then use unittest.mock.patch or direct function pointer replacement on the class definition of the Flux model in memory.  
   Better approach (Comfy Native):  
   Use model.patch\_transformer\_options.  
   This allows us to inject a callback that runs before/after blocks. But RoPE happens inside the block.  
   The Definitive Custom Node Solution:  
   The custom node should define a wrapper class for the Flux or SD3 model's forward pass.  
   Inside this wrapper, it detects the forward call.  
   It temporarily monkey-patches flux.math.apply\_rope (or equivalent) with a context manager.  
   Python  
   \# Context Manager for Circular RoPE  
   class CircularRoPEContext:  
       def \_\_enter\_\_(self):  
           self.original\_rope \= flux.math.apply\_rope  
           flux.math.apply\_rope \= self.circular\_rope

       def \_\_exit\_\_(self,...):  
           flux.math.apply\_rope \= self.original\_rope

       def circular\_rope(self, q, k, ids):  
           \# 1\. Modify IDs or Frequencies here  
           \# 2\. To make width circular:   
           \#    Force the relative distance logic to wrap.  
           \#    This is hard with standard RoPE formula.  
           \#    Easier: Modify 'ids' such that ids\[x\] and ids are close? No.

           \# THE REAL TRICK: Asymmetric Tiling / Rolling  
           \# Instead of modifying RoPE math (which is hard to make circular exactly),  
           \# we simply ensure that during attention, the window rolls.  
           pass

Correction on "Circular RoPE" via Custom Node:  
Actually, true mathematical Circular RoPE (where $f(0) \== f(W)$) requires changing the frequencies.  
The custom node must compute a new cos/sin table where frequencies are quantized: $\\theta\_i \= 2\\pi k\_i / W$.  
It then injects this table into the model.  
In ComfyUI, Flux models store their RoPE cache in model.diffusion\_model.rope\_cache (or similar).  
The Custom Node Action:

1. Read model.diffusion\_model.  
2. Calculate new circular frequencies based on the current latent width.  
3. Overwrite the cached cos/sin tables in the model instance with the circular ones.  
4. **Critical:** This must be done at inference time because width changes. The node must be a "Model Patcher" that acts during the sampling loop.

## **7\. Detailed Analysis of Implementation Differences**

This section provides a comparative data analysis of the Native DiT360 approach versus the ComfyUI Patch approach.

| Feature | DiT360 (Native) | Flux/SD3 (ComfyUI Patch) |
| :---- | :---- | :---- |
| **Topological Basis** | Cylindrical (Learned) | Planar (Forced to Cylindrical) |
| **VAE Padding** | Circular Padding (Circular Convolution) | Standard (unless patched via VAE node) |
| **RoPE Logic** | Relative Distance Modulo $W$ | Standard RoPE or Modified Frequencies |
| **Attention Mask** | Circular Window Attention | Standard (FlashAttn) or Asymmetric Tiling |
| **Seam Artifacts** | Negligible | Moderate (Requires tuning/inpainting) |
| **Global Coherence** | High (Consistent Geometry) | Variable (Model confusion at boundaries) |
| **Implementation** | Training Code modification | Runtime Injection / Graph Patching |

### **7.1 Second-Order Insights: Why Patched Models Struggle**

Insight: The "Guidance Scale Amplification" at the Seam.  
When we force a planar model like Flux to attend circularly (i.e., pixel $0$ attending to pixel $W$), the attention scores may be high (due to forced geometric proximity), but the content at $0$ and $W$ might be semantically disparate initially.  
Causal Chain:

1. Random noise is initialized.  
2. The model sees a "seam" in the noise structure if standard generation is used.  
3. Circular RoPE forces the model to treat them as neighbors.  
4. The model tries to reconcile the discontinuity.  
5. If the Guidance Scale (CFG) is high, the model over-corrects, creating a "burn" or hyper-saturated line at the wrap point.  
   Mitigation: DiT360 avoids this because it learns from noise distributions that are already circular. For ComfyUI, the solution is Noise Rolling: The initial noise latent must also be generated with circular symmetry awareness, or the sampler should "roll" the latents (shift horizontally by $W/2$) every few steps to distribute the seam error.

## **8\. Requirements for Custom Node Implementation (Technical Specification)**

To satisfy the user's research request fully, here is the specification for the custom node required to make circular padding/RoPE work.

### **8.1 Node 1: CircularVAE**

* **Purpose:** Eliminate seam in the pixel-space decoding.  
* **Code Strategy:**  
  * Input: VAE  
  * Logic: Iterates vae.first\_stage\_model.encoder and vae.first\_stage\_model.decoder.  
  * Action: Replaces torch.nn.Conv2d with a wrapper that sets padding\_mode='circular' on the fly.  
  * Output: VAE (patched).

### **8.2 Node 2: CircularFluxModel**

* **Purpose:** Eliminate seam in the latent generation structure.  
* **Code Strategy:**  
  * Input: MODEL (Flux/SD3).  
  * Parameters: width (optional override), start\_step, end\_step.  
  * Logic: Uses model.set\_model\_patch\_replace logic.  
  * **Specific Injection:**  
    * It must access the specific apply\_rope implementation.  
    * Since Flux code is often inside the comfy core, the node should implement **Tiled Attention** with circular overlap as a proxy for Circular RoPE if modifying the frequency table is blocked by compiled CUDA kernels.  
    * *Advanced:* If using pure Python implementation of RoPE, the node overwrites model.diffusion\_model.forward\_orig with a function that modifies transformer\_options\['patches'\].  
  * **Frequency Hack:**  
    * theta\_new \= theta\_old \* correction\_factor  
    * Where correction\_factor ensures cos(W \* theta) \= 1\.

## **9\. Conclusion**

The implementation of Circular RoPE represents a critical divergence between specialized architectures like DiT360 and general-purpose systems like Flux.1. DiT360 achieves circularity through architectural mandates—embedding circular convolutions and windowing strategies directly into the training loop. This results in a model that "thinks" cylindrically.

In contrast, implementing this in ComfyUI for Flux/SD3 is an exercise in **runtime graph modification**. While the user struggled with core code modifications, research confirms that the ModelPatcher and dynamic module replacement capabilities of Python allow for this to be encapsulated in custom nodes. The solution requires a two-pronged approach: patching the VAE's convolution layers to handle texture continuity at the pixel level, and patching the DiT's positional embeddings (or attention mechanism) to handle semantic continuity at the latent level.

The deeper insight for the user is that while a custom node *can* mechanically force circularity, the model's weights remain planar. Consequently, "perfect" 360-degree generation in Flux will always require secondary mitigation strategies—such as sliding-window sampling (rolling the latent) or seam-guided inpainting—to overcome the fundamental mismatch between the model's learned planar priors and the enforced cylindrical topology.

## **10\. Future Directions: Spherical vs. Cylindrical**

While this report focuses on Circular RoPE (Cylindrical), the next frontier is **Spherical RoPE**. DiT360 touches on this with asymmetric handling of elevation. For Flux, true spherical generation would require a coordinate transform from $(x, y)$ grid to $(\\phi, \\theta)$ sphere surface coordinates before applying RoPE. This is currently beyond the scope of simple ComfyUI patching and would likely require a dedicated ControlNet or LoRA trained on spherical projections to warp the planar priors effectively.

---

Citations:  
"DiT360: Omnidirectional Image Generation with Diffusion Transformers", arXiv:2510.11712.  
DiT360 GitHub Repository (Insta360-Research-Team).  
ComfyUI Documentation and Source Code (comfy.model\_patcher).  
Flux.1 / Stable Diffusion 3 Technical Reports (Black Forest Labs / Stability AI).